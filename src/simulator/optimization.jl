
export update_objective_new_parameters!, setup_parameter_optimization, optimization_config

function update_objective_new_parameters!(param_serialized, sim, state0, param, forces, dt, G; log_obj = false, config = nothing, kwarg...)
    if isnothing(config)
        config = simulator_config(sim; kwarg...)
    end
    devectorize_variables!(param, sim.model, param_serialized, :parameters)
    states, = simulate(state0, sim, dt, parameters = param, forces = forces, config = config)
    obj = evaluate_objective(G, sim.model, states, dt, forces)
    if log_obj
        println("Current objective: $obj")
    end
    return (obj, states)
end

"""
    setup_parameter_optimization(model, state0, param, dt, forces, G, opt_cfg = optimization_config(model, param);
                                                            grad_type = :adjoint,
                                                            config = nothing,
                                                            print = 1,
                                                            copy_case = true,
                                                            param_obj = false,
                                                            kwarg...)

Set up function handles for optimizing the case defined by the inputs to
`simulate` together with a per-timestep objective function `G`.

Generally calling either of the functions will mutate the data Dict. The options are:
F_o(x) -> evaluate objective
dF_o(dFdx, x) -> evaluate gradient of objective, mutating dFdx (may trigger evaluation of F_o)
F_and_dF(F, dFdx, x) -> evaluate F and/or dF. Value of nothing will mean that the corresponding entry is skipped.

"""
function setup_parameter_optimization(model, state0, param, dt, forces, G, arg...; kwarg...)
    case = JutulCase(model, dt, forces, state0 = state0, parameters = param)
    return setup_parameter_optimization(case, G, arg...; kwarg...)
end

function setup_parameter_optimization(case::JutulCase, G, opt_cfg = optimization_config(case.model, case.parameters);
        grad_type = :adjoint,
        config = nothing,
        print = 1,
        copy_case = true,
        param_obj = false,
        use_sparsity = true,
        kwarg...
    )
    if copy_case
        case = duplicate(case)
    end
    # Pick active set of targets from the optimization config and construct a mapper
    (; model, state0, parameters) = case
    if print isa Bool
        if print
            print = 1
        else
            print = Inf
        end
    end
    verbose = print > 0 && isfinite(print)
    targets = optimization_targets(opt_cfg, model)
    if grad_type == :numeric
        @assert length(targets) == 1
        @assert model isa SimulationModel
    else
        @assert grad_type == :adjoint
    end
    sort_variables!(model, :parameters)
    mapper, = variable_mapper(model, :parameters, targets = targets, config = opt_cfg)
    lims = optimization_limits(opt_cfg, mapper, parameters, model)
    if verbose
        print_parameter_optimization_config(targets, opt_cfg, model)
    end
    x0 = vectorize_variables(model, parameters, mapper, config = opt_cfg)
    for k in eachindex(x0)
        low = lims[1][k]
        high = lims[2][k]
        @assert low <= x0[k] "Computed lower limit $low for parameter #$k was larger than provided x0[k]=$(x0[k])"
        @assert high >= x0[k] "Computer upper limit $high for parameter #$k was smaller than provided x0[k]=$(x0[k])"
    end
    data = Dict()
    data[:n_objective] = 1
    data[:n_gradient] = 1
    data[:obj_hist] = zeros(0)

    sim = Simulator(case)
    if isnothing(config)
        config = simulator_config(sim; info_level = -1, kwarg...)
    elseif !verbose
        config[:info_level] = -1
        config[:end_report] = false
    end
    data[:sim] = sim
    data[:sim_config] = config

    if grad_type == :adjoint
        adj_storage = setup_adjoint_storage(model,
            state0 = state0,
            parameters = parameters,
            targets = targets,
            use_sparsity = use_sparsity,
            param_obj = param_obj
        )
        data[:adjoint_storage] = adj_storage
        grad_adj = zeros(adj_storage.n)
    else
        grad_adj = similar(x0)
    end
    data[:case] = case
    data[:grad_adj] = grad_adj
    data[:mapper] = mapper
    data[:G] = G
    data[:targets] = targets
    data[:mapper] = mapper
    data[:config] = opt_cfg
    data[:last_obj] = Inf
    data[:best_obj] = Inf
    data[:best_x] = copy(x0)
    data[:x_hash] = hash(Inf)
    F = x -> objective_opt!(x, data, print)
    dF = (dFdx, x) -> gradient_opt!(dFdx, x, data)
    F_and_dF = (F, dFdx, x) -> objective_and_gradient_opt!(F, dFdx, x, data, print)
    return (F! = F, dF! = dF, F_and_dF! = F_and_dF, x0 = x0, limits = lims, data = data)
end


function gradient_opt!(dFdx, x, data)
    (; state0, parameters, forces, dt) = data[:case]
    G = data[:G]
    targets = data[:targets]
    mapper = data[:mapper]
    opt_cfg = data[:config]
    model = data[:sim].model

    data[:n_gradient] += 1
    grad_adj = data[:grad_adj]
    if haskey(data, :adjoint_storage)
        states = data[:states]
        if length(states) == length(dt)
            storage = data[:adjoint_storage]
            for sim in [storage.forward, storage.backward, storage.parameter]
                for k in [:state, :state0, :parameters]
                    reset_variables!(sim, parameters, type = k)
                end
            end
            debug_time = false
            set_global_timer!(debug_time)
            try
                grad_adj = solve_adjoint_sensitivities!(grad_adj, storage, states, state0, dt, G, forces = forces)
            catch excpt
                @warn "Exception in adjoint solve, setting gradient to large value." excpt
                @. grad_adj = 1e10
            end
            print_global_timer(debug_time; text = "Adjoint solve detailed timing")
        else
            @. grad_adj = 1e10
        end
    else
        grad_adj = Jutul.solve_numerical_sensitivities(
            model, data[:states], data[:reports], G, only(targets),
            state0 = state0,
            forces = forces,
            parameters = parameters
        )
    end
    @. dFdx = 0.0
    transfer_gradient!(dFdx, grad_adj, x, mapper, opt_cfg, model)
    @assert all(isfinite, dFdx) "Non-finite gradients detected."
    return dFdx
end

function objective_opt!(x, data, print_frequency = 1)
    (; state0, parameters, forces, dt) = data[:case]
    # state0 = data[:state0]
    # param = data[:parameters]
    # dt = data[:dt]
    # forces = data[:forces]
    G = data[:G]
    mapper = data[:mapper]
    opt_cfg = data[:config]
    sim = data[:sim]
    model = sim.model
    devectorize_variables!(parameters, model, x, mapper, config = opt_cfg)
    config = data[:sim_config]
    states, reports = simulate(state0, sim, dt, parameters = parameters, forces = forces, config = config)
    data[:states] = states
    data[:reports] = reports
    bad_obj = 10*data[:last_obj]
    obj = evaluate_objective(G, sim.model, states, dt, forces, large_value = bad_obj)
    data[:x_hash] = hash(x)
    n = data[:n_objective]
    push!(data[:obj_hist], obj)
    if print_frequency > 0 && mod(n, print_frequency) == 0
        fmt = x -> @sprintf("%2.4e", x)
        rel = obj/data[:obj_hist][1]
        best = data[:best_obj]
        jutul_message("Obj. #$n", "$(fmt(obj)) (best: $(fmt(best)), relative: $(fmt(rel)))")
    end
    data[:n_objective] += 1
    if obj != bad_obj
        data[:last_obj] = obj
    end
    if obj < data[:best_obj]
        data[:best_obj] = obj
        data[:best_x] .= x
    end
    return obj
end

function objective_and_gradient_opt!(F, dFdx, x, data, arg...)
    last_obj = data[:last_obj]
    # Might ask for one or the other
    want_grad = !isnothing(dFdx)
    want_obj = !isnothing(F)
    hash_mismatch = data[:x_hash] != hash(x)
    need_recompute_obj = hash_mismatch || !isfinite(last_obj)
    if need_recompute_obj
        # Adjoint only valid if objective has been computed for current x
        objective_opt!(x, data, arg...)
    end
    if want_obj
        # Might be updated or might be last_obj, get it anyway.
        obj = data[:last_obj]
    else
        obj = nothing
    end
    if want_grad
        gradient_opt!(dFdx, x, data)
    end
    return obj
end

function optimization_config(case::JutulCase, active = parameter_targets(case.model); kwarg...)
    model = case.model
    param = case.parameters
    return optimization_config(model, param, active; kwarg...)
end

function optimization_config(model::SimulationModel, param, active = parameter_targets(model);
        rel_min = nothing,
        rel_max = nothing,
        use_scaling = false
    )
    out = Dict{Symbol, Any}()
    for k in active
        var = model.parameters[k]
        scale = variable_scale(var)
        if isnothing(scale)
            scale = 1.0
        end
        abs_min = minimum_value(var)
        if isnothing(abs_min)
            abs_min = -Inf
        end
        abs_max = maximum_value(var)
        if isnothing(abs_max)
            abs_max = Inf
        end
        out[k] = OrderedDict(
            :active => true,
            :use_scaling => use_scaling,
            :scaler => :default,
            :abs_min => abs_min,
            :abs_max => abs_max,
            :rel_min => rel_min,
            :rel_max => rel_max,
            :base_scale => scale,
            :lumping => nothing,
            :low => nothing,
            :high => nothing
        )
    end
    return out
end

function optimization_targets(config::Dict, model)
    out = Vector{Symbol}()
    for (k, v) in pairs(config)
        if v[:active]
            push!(out, k)
        end
    end
    return out
end

opt_scaler_function(config::Nothing, key; inv = false) = x -> x

function opt_scaler_function(config, key; inv = false)
    cfg = config[key]
    scale_type = cfg[:scaler]
    if cfg[:use_scaling]
        x_min = cfg[:low]
        x_max = cfg[:high]
        if isnothing(x_min)
            x_min = 0.0
        end
        if isnothing(x_max)
            # Divide by base scaling if no max value
            Δ = cfg[:base_scale]
        else
            Δ = x_max - x_min
        end
        F = F_inv = identity

        if scale_type == :log || scale_type == :exp
            if x_max ≈ x_min || x_min < 1e-12
                base = 10000
            else
                base = abs(x_max)/abs(x_min)
            end
            myexp = x -> (base^x - 1)/(base - 1)
            mylog = x -> log((base-1)*x + 1)/log(base)
            if scale_type == :log
                F_inv, F = myexp, mylog
            else
                F, F_inv = myexp, mylog
            end
        else
            @assert scale_type == :default "Unknown scaler $scale_type"
        end
        if inv
            scaler = x -> F_inv(x)*Δ + x_min
        else
            scaler = x -> F((x - x_min)/Δ)
        end
    else
        if scale_type == :default
            scaler = identity
        elseif scale_type == :exp
            scaler = inv ? log : exp
        elseif scale_type == :log
            scaler = inv ? exp : log
        else
            error("Unknown scaler $scale_type")
        end
    end
    return scaler
end

function optimization_limits(config, mapper, param, model)
    n = vectorized_length(model, mapper)
    x_min = zeros(n)
    x_max = zeros(n)
    lims = (min = x_min, max = x_max)
    lims = optimization_limits!(lims, config, mapper, param, model)
    return lims
end

function optimization_limits!(lims, config, mapper, param, model)
    x_min, x_max = lims
    for (param_k, v) in mapper
        (; n_full, n_x, offset_full, offset_x) = v
        cfg = config[param_k]
        vals = param[param_k]

        rel_max = cfg[:rel_max]
        rel_min = cfg[:rel_min]
        # We have found limits in terms of unscaled variable, scale first
        abs_max = cfg[:abs_max]
        abs_min = cfg[:abs_min]
        low_group = Inf
        high_group = -Inf
        for i in 1:n_x
            k = i + offset_x
            val = vals[i]
            if isnothing(rel_min)
                low = abs_min
            else
                if val < 0
                    rel_min_actual = val/rel_min
                else
                    rel_min_actual = val*rel_min
                end
                low = max(abs_min, rel_min_actual)
            end
            if isnothing(rel_max)
                hi = abs_max
            else
                if val < 0
                    rel_max_actual = val/rel_max
                else
                    rel_max_actual = val*rel_max
                end
                hi = min(abs_max, rel_max_actual)
            end
            @assert !isnan(low)
            @assert !isnan(hi)
            x_min[k] = low
            x_max[k] = hi

            low_group = min(low_group, low)
            high_group = max(high_group, hi)
        end
        if high_group != Inf
            high_group = max(high_group, low_group + 1e-8*(low_group + high_group) + 1e-18)
        end
        @assert !isnan(low_group)
        @assert !isnan(high_group)
        cfg[:low] = low_group
        cfg[:high] = high_group
        cscale = (low_group, high_group)

        F = opt_scaler_function(config, param_k, inv = false)
        # F_inv = opt_scaler_function(config, k, inv = true)
        for i in 1:n_x
            k = i + offset_x
            low = F(x_min[k])
            hi = F(x_max[k])
            @assert !isnan(low) "Scaled limit for F($(x_min[k])) was NaN (urange $cscale)"
            @assert !isnan(hi) "Scaled limit for F($(x_max[k])) was NaN (urange $cscale)"
            x_min[k] = low
            x_max[k] = hi
        end
        lumping = get_lumping(cfg)
        if !isnothing(lumping)
            if vals isa AbstractVector
                for lno in 1:maximum(lumping)
                    pos = findfirst(isequal(lno), lumping)
                    ref_val = vals[pos]
                    for (i, l) in enumerate(lumping)
                        if l == lno
                            if !(vals[i] ≈ ref_val)
                                error("Initial values for $param_k differed for lumping group $lno at position $i")
                            end
                        end
                    end
                end
            else
                for lno in 1:maximum(lumping)
                    pos = findfirst(isequal(lno), lumping)
                    ref_val = vals[:, pos]
                    for (i, l) in enumerate(lumping)
                        if l == lno
                            if !(vals[:, i] ≈ ref_val)
                                error("Initial values for $param_k differed for lumping group $lno at position $i")
                            end
                        end
                    end
                end
            end
        end
    end
    return lims
end

function transfer_gradient!(dGdy, dGdx, y, mapper, config, model)
    # Note: dGdy is a vector of short length (i.e. with lumping) and dGdx is a
    # vector of full length (i.e. without lumping)
    for (varname, v) in mapper
        (; n_full, n_x, offset_full, offset_x, n_row) = v
        lumping = get_lumping(config[varname])
        x_to_y = opt_scaler_function(config, varname, inv = false)
        y_to_x = opt_scaler_function(config, varname, inv = true)

        if isnothing(lumping)
            @assert n_x == n_full
            for i in 1:n_x
                k = offset_x + i
                k_full = offset_full + i
                dGdy[k] = objective_gradient_chain_rule(x_to_y, y_to_x, y[k], dGdx[k_full])
            end
        else
            lumping::AbstractVector
            m_x = n_x ÷ n_row
            m_full = n_full ÷ n_row
            @assert m_x == maximum(lumping) "Lumping group $k has $m_x groups, but $n_x variables"
            indx(j, lump) = offset_x + (lump - 1)*n_row + j
            for lump in 1:m_x
                for j in 1:n_row
                    dGdy[indx(j, lump)] = 0.0
                end
            end
            for (i, lump) in enumerate(lumping)
                for j in 1:n_row
                    ix = indx(j, lump)
                    # Note: Gradients follow canonical order (equation major)
                    ix_full = offset_full + (j - 1)*m_full + i
                    dGdy[ix] += objective_gradient_chain_rule(x_to_y, y_to_x, y[ix], dGdx[ix_full])
                end
            end
        end
    end
    return dGdy
end

function objective_gradient_chain_rule(x_to_y, y_to_x, y, dGdx)
    x = y_to_x(y)
    x_ad = ForwardDiff.Dual(x, 1.0)
    y_ad = x_to_y(x_ad)
    dydx = only(y_ad.partials)
    # dG(y(x))/dx = dG/dy * dy/dx
    # -> dG/dy = dG/dx / dy/dx
    # The following is fine as dydx should never be zero
    dGdy = dGdx/dydx
    return dGdy
end

function print_parameter_optimization_config(targets, config, model; title = :model)
    nt = length(targets)
    if nt > 0
        data = Matrix{Any}(undef, nt, 8)
        for (i, target) in enumerate(targets)
            prm = model.parameters[target]
            e = associated_entity(prm)
            n = count_active_entities(model.domain, e)
            m = degrees_of_freedom_per_entity(model, prm)
            v = config[target]
            data[i, 1] = target
            data[i, 2] = "$e"[1:end-2]
            if m == 1
                s = "$n"
            else
                s = "$n×$m=$(n*m)"
            end
            lumping = get_lumping(v)
            if isnothing(lumping)
                lstr = "-"
            else
                n = length(unique(lumping))
                estr = "($n×$m=$(n*m) dof)"
                if n == 1
                    lstr = "1 group $estr"
                else
                    lstr = "$n groups $estr"
                end
            end
            function fmt_lim(l, u)
                if isnothing(l)
                    l = -Inf
                end
                if isnothing(u)
                    u = Inf
                end
                @sprintf "[%1.3g, %1.3g]" l u
            end
            data[i, 3] = s
            data[i, 4] = v[:scaler]
            data[i, 5] = fmt_lim(v[:abs_min], v[:abs_max])
            data[i, 6] = fmt_lim(v[:rel_min], v[:rel_max])
            data[i, 7] = fmt_lim(v[:low], v[:high])
            data[i, 8] = lstr
        end
        h = ["Name", "Entity", "N", "Scale", "Abs. limits", "Rel. limits", "Limits", "Lumping"]
        pretty_table(data, header = h, title = "Parameters for $title")
    end
end
