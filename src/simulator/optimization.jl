
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

function setup_parameter_optimization(model, state0, param, dt, forces, G, opt_cfg = optimization_config(model, param);
                                                            grad_type = :adjoint,
                                                            config = nothing,
                                                            print = 1,
                                                            copy_parameters = true,
                                                            param_obj = false,
                                                            kwarg...)
    # Pick active set of targets from the optimization config and construct a mapper
    if print isa Bool
        if print
            print = 1
        else
            print = Inf
        end
    end
    if copy_parameters
        param = deepcopy(param)
    end
    targets = optimization_targets(opt_cfg, model)
    if grad_type == :numeric
        @assert length(targets) == 1
        @assert model isa SimulationModel
    else
        @assert grad_type == :adjoint
    end
    mapper, = variable_mapper(model, :parameters, targets = targets, config = opt_cfg)
    lims = optimization_limits(opt_cfg, mapper, param, model)
    if print > 0
        print_parameter_optimization_config(targets, opt_cfg, model)
    end
    x0 = vectorize_variables(model, param, mapper, config = opt_cfg)
    for k in eachindex(x0)
        low = lims[1][k]
        high = lims[2][k]
        @assert low <= x0[k] "Computed lower limit $low for parameter #$k was larger than provided x0[k]=$(x0[k])"
        @assert high >= x0[k] "Computer upper limit $hi for parameter #$k was smaller than provided x0[k]=$(x0[k])"
    end
    data = Dict()
    data[:n_objective] = 1
    data[:n_gradient] = 1
    data[:obj_hist] = zeros(0)

    sim = Simulator(model, state0 = state0, parameters = param)
    if isnothing(config)
        config = simulator_config(sim; info_level = -1, kwarg...)
    end
    data[:sim] = sim
    data[:sim_config] = config

    # grad_adj = similar(x0)
    # error()
    if grad_type == :adjoint
        adj_storage = setup_adjoint_storage(model, state0 = state0,
                                                   parameters = param,
                                                   targets = targets,
                                                   param_obj = param_obj)
        data[:adjoint_storage] = adj_storage
        grad_adj = zeros(adj_storage.n)
    else
        grad_adj = similar(x0)
    end
    data[:grad_adj] = grad_adj
    data[:mapper] = mapper
    data[:dt] = dt
    data[:forces] = forces
    data[:parameters] = param
    data[:G] = G
    data[:targets] = targets
    data[:mapper] = mapper
    data[:config] = opt_cfg
    data[:state0] = state0
    data[:last_obj] = Inf
    F = x -> objective_opt!(x, data, print)
    dF = (dFdx, x) -> gradient_opt!(dFdx, x, data)
    F_and_dF = (F, dFdx, x) -> objective_and_gradient_opt!(F, dFdx, x, data, print)
    return (F! = F, dF! = dF, F_and_dF! = F_and_dF, x0 = x0, limits = lims, data = data)
end

function gradient_opt!(dFdx, x, data)
    state0 = data[:state0]
    param = data[:parameters]
    dt = data[:dt]
    forces = data[:forces]
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
                    reset_variables!(sim, param, type = k)
                end
            end
            debug_time = false
            set_global_timer!(debug_time)
            grad_adj = solve_adjoint_sensitivities!(grad_adj, storage, states, state0, dt, G, forces = forces)
            print_global_timer(debug_time; text = "Adjoint solve detailed timing")
        else
            @. grad_adj = 1e10
        end
    else
        grad_adj = Jutul.solve_numerical_sensitivities(model, data[:states], data[:reports], G, only(targets),
                                                                    state0 = state0, forces = forces, parameters = param)
    end
    transfer_gradient!(dFdx, grad_adj, x, mapper, opt_cfg, model)
    @assert all(isfinite, dFdx) "Non-finite gradients detected."
    return dFdx
end

function objective_opt!(x, data, print_frequency = 1)
    state0 = data[:state0]
    param = data[:parameters]
    dt = data[:dt]
    forces = data[:forces]
    G = data[:G]
    mapper = data[:mapper]
    opt_cfg = data[:config]
    sim = data[:sim]
    model = sim.model
    devectorize_variables!(param, model, x, mapper, config = opt_cfg)
    config = data[:sim_config]
    states, reports = simulate(state0, sim, dt, parameters = param, forces = forces, config = config)
    data[:states] = states
    data[:reports] = reports
    bad_obj = 10*data[:last_obj]
    obj = evaluate_objective(G, sim.model, states, dt, forces, large_value = bad_obj)
    data[:n_objective] += 1
    n = data[:n_objective]
    push!(data[:obj_hist], obj)
    if mod(n, print_frequency) == 0
        println("#$n: $obj")
    end
    if obj != bad_obj
        data[:last_obj] = obj
    end
    return obj
end

function objective_and_gradient_opt!(F, dFdx, x, data, arg...)
    obj = objective_opt!(x, data, arg...)
    if !isnothing(dFdx)
        gradient_opt!(dFdx, x, data)
    end
    return obj
end

function optimization_config(model, param, active = keys(model.parameters);
                                                        rel_min = nothing,
                                                        rel_max = nothing,
                                                        use_scaling = false)
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
            base = 10000.0
            myexp = x -> (base^x - 1)/(base - 1)
            mylog = x -> log((base-1)*x + 1)/log(base)
            if scale_type == :exp
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
        elseif scale_type == :log
            scaler = inv ? log : exp
        elseif scale_type == :exp
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
        (; offset, n) = v
        cfg = config[param_k]
        vals = param[param_k]

        rel_max = cfg[:rel_max]
        rel_min = cfg[:rel_min]
        # We have found limits in terms of unscaled variable, scale first
        abs_max = cfg[:abs_max]
        abs_min = cfg[:abs_min]
        low_group = Inf
        high_group = -Inf
        for i in 1:n
            k = i + offset
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
            x_min[k] = low
            x_max[k] = hi

            low_group = min(low_group, low)
            high_group = max(high_group, hi)
        end
        high_group = max(high_group, low_group + 1e-8*(low_group + high_group))
        cfg[:low] = low_group
        cfg[:high] = high_group

        F = opt_scaler_function(config, param_k, inv = false)
        # F_inv = opt_scaler_function(config, k, inv = true)
        for i in 1:n
            k = i + offset
            x_min[k] = F(x_min[k])
            x_max[k] = F(x_max[k])
        end
        lumping = get_lumping(cfg)
        if !isnothing(lumping)
            for lno in 1:maximum(lumping)
                pos = findfirst(isequal(lno), lumping)
                ref_val = vals[pos + offset]
                for (i, l) in enumerate(lumping)
                    if l == lno
                        if vals[i + offset] != ref_val
                            error("Initial values for $param_k differed for lumping group $lno at position $i")
                        end
                    end
                end
            end
        end
    end
    return lims
end

function transfer_gradient!(dGdy, dGdx, y, mapper, config, model)
    for (k, v) in mapper
        (; offset, n) = v
        x_to_y = opt_scaler_function(config, k, inv = false)
        y_to_x = opt_scaler_function(config, k, inv = true)
        for i in 1:n
            k = offset + i
            dGdy[k] = objective_gradient_chain_rule(x_to_y, y_to_x, y[k], dGdx[k])
        end
    end
    return dGdy
end

function objective_gradient_chain_rule(x_to_y, y_to_x, y, dGdx)
    x = y_to_x(y)
    x_ad = ForwardDiff.Dual(x, 1.0)
    y_ad = x_to_y(x_ad)
    dydx = only(y_ad.partials)
    # dG(y(x))/dx = dG/dx * dy/dx
    dGdy = dGdx*dydx
    return dGdy
end

function print_parameter_optimization_config(targets, config, model; title = :model)
    nt = length(targets)
    if nt > 0
        data = Matrix{Any}(undef, nt, 7)
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
            data[i, 3] = s
            data[i, 4] = v[:scaler]
            data[i, 5] = (v[:abs_min], v[:abs_max])
            data[i, 6] = (v[:rel_min], v[:rel_max])
            data[i, 7] = (v[:low], v[:high])
        end
        h = ["Name", "Entity", "N", "Scale", "Abs. limits", "Rel. limits", "Limits"]
        pretty_table(data, header = h, title = "Parameters for $title")
    end
end
