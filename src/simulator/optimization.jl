
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
                                                            print_obj = true,
                                                            copy_parameters = true,
                                                            param_obj = false,
                                                            kwarg...)
    # Pick active set of targets from the optimization config and construct a mapper
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
    mapper, = variable_mapper(model, :parameters, targets = targets)
    x0 = vectorize_variables(model, param, mapper, config = opt_cfg)
    data = Dict()
    data[:n_objective] = 1
    data[:n_gradient] = 1

    sim = Simulator(model, state0 = state0, parameters = param)
    if isnothing(config)
        config = simulator_config(sim; info_level = -1, kwarg...)
    end

    function F(x)
        devectorize_variables!(param, sim.model, x, mapper, config = opt_cfg)
        states, reports = simulate(state0, sim, dt, parameters = param, forces = forces, config = config)
        data[:states] = states
        data[:reports] = reports
        obj = evaluate_objective(G, sim.model, states, dt, forces)
        data[:n_objective] += 1
        n = data[:n_objective]
        if print_obj
            println("#$n: $obj")
        end
        return obj
    end
    grad_adj = similar(x0)
    data[:grad_adj] = grad_adj
    if grad_type == :adjoint
        data[:adjoint_storage] = setup_adjoint_storage(model, state0 = state0, parameters = param, targets = targets, param_obj = param_obj)
    else
        @assert grad_type == :numeric
    end
    lims = optimization_limits(opt_cfg, mapper, x0, param, model)
    data[:mapper] = mapper
    data[:config] = opt_cfg
    dF = (dFdx, x) -> dFdx!(dFdx, x, data, model, state0, param, dt, forces, G, targets, mapper, opt_cfg)
    return (F, dF, x0, lims, data)
end

function dFdx!(dFdx, x, data, model, state0, param, dt, forces, G, targets, mapper, opt_cfg)
    data[:n_gradient] += 1
    grad_adj = data[:grad_adj]
    if haskey(data, :adjoint_storage)
        storage = data[:adjoint_storage]
        for sim in [storage.forward, storage.backward, storage.parameter]
            for k in [:state, :state0, :parameters]
                reset_variables!(sim, param, type = k)
            end
        end
        grad_adj = solve_adjoint_sensitivities!(grad_adj, storage, data[:states], state0, dt, G, forces = forces)
    else
        grad_adj = Jutul.solve_numerical_sensitivities(model, data[:states], data[:reports], G, only(targets),
                                                                    state0 = state0, forces = forces, parameters = param)
    end
    transfer_gradient!(dFdx, grad_adj, x, mapper, opt_cfg, model)
    @assert all(isfinite, dFdx) "Non-finite gradients detected."
    return dFdx
end



function optimization_config(model, param, active = keys(model.parameters);
                                                        rel_min = nothing,
                                                        rel_max = nothing,
                                                        use_scaling = true)
    out = Dict{Symbol, Any}()
    ϵ = 1e-8
    for k in active
        var = model.parameters[k]
        vals = param[k]
        scale = variable_scale(var)
        if isnothing(scale)
            scale = 1.0
        end
        # Low/high is not bounds, but typical scaling values
        K = 10
        low = minimum(vec(vals))/K - ϵ*scale
        hi = maximum(vec(vals))*K + ϵ*scale
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
                            :low => low,
                            :high => hi
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
    if cfg[:use_scaling]
        scale_type = cfg[:scaler]
        if scale_type == :default
            x_min = cfg[:low]
            x_max = cfg[:high]
            if isnothing(x_min)
                x_min = 0.0
            end
            if isnothing(x_max)
                # Divide by 1.0 if no max value
                Δ = cfg[:base_scale]
            else
                Δ = x_max - x_min
            end
            if inv
                scaler = x -> x*Δ + x_min
            else
                scaler = x -> (x - x_min)/Δ
            end
        elseif scale_type == :log
            scaler = inv ? exp : log
        else
            error("Unknown scaler $scale_type")
        end
    else
        scaler = identity
    end
    return scaler
end

function optimization_limits(config, mapper, x0, param, model)
    x_min = similar(x0)
    x_max = similar(x0)
    lims = (min = x_min, max = x_max)
    lims = optimization_limits!(lims, config, mapper, x0, param, model)
    return lims
end

function optimization_limits!(lims, config, mapper, x0, param, model)
    x_min, x_max = lims
    for (k, v) in mapper
        (; offset, n) = v
        cfg = config[k]
        vals = param[k]
        F = opt_scaler_function(config, k, inv = false)
        # F_inv = opt_scaler_function(config, k, inv = true)

        rel_max = cfg[:rel_max]
        rel_min = cfg[:rel_min]
        # We have found limits in terms of unscaled variable, scale first
        abs_max = F(cfg[:abs_max])
        abs_min = F(cfg[:abs_min])
        for i in 1:n
            k = i + offset
            val = F(vals[i])
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
            @assert low <= x0[k] "Computed lower limit $low for parameter #$k was larger than provided x0[k]=$(x0[k])"
            @assert hi >= x0[k] "Computer upper limit $hi for parameter #$k was smaller than provided x0[k]=$(x0[k])"
            x_min[k] = low
            x_max[k] = hi
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
