
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
                                                            grad_type = :adjoint, config = nothing, kwarg...)
    # Pick active set of targets from the optimization config and construct a mapper
    targets = optimization_targets(opt_cfg, model)
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
        states, = simulate(state0, sim, dt, parameters = param, forces = forces, config = config)
        data[:states] = states
        obj = evaluate_objective(G, sim.model, states, dt, forces)
        data[:n_objective] += 1
        n = data[:n_objective]
        println("#$n: $obj")
        return obj
    end
    @assert grad_type == :adjoint
    grad_adj = similar(x0)
    data[:grad_adj] = grad_adj
    function dF(dFdx, x)
        # TODO: Avoid re-allocating storage.
        devectorize_variables!(param, model, x, mapper, config = opt_cfg)
        storage = setup_adjoint_storage(model, state0 = state0, parameters = param)
        grad_adj = solve_adjoint_sensitivities!(grad_adj, storage, data[:states], state0, dt, G, forces = forces)
        data[:n_gradient] += 1
        # TODO: Perform scaling of gradients here.
        dFdx .= grad_adj
        return dFdx
    end
    lims = optimization_limits(opt_cfg, mapper, x0, param, model)
    return (F, dF, x0, lims, data)
end

function optimization_config(model, param, active = keys(model.parameters); rel_min = nothing, rel_max = nothing)
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
        low = minimum(vec(vals)) - ϵ*scale
        hi = maximum(vec(vals)) + ϵ*scale
        abs_min = minimum_value(var)
        if isnothing(abs_min)
            abs_min = -Inf
        end
        abs_max = maximum_value(var)
        if isnothing(abs_max)
            abs_max = Inf
        end
        out[k] = Dict(:active => true,
                      :abs_min => abs_min,
                      :abs_max => abs_max,
                      :rel_min => rel_min,
                      :rel_max => rel_max,
                      :low => low,
                      :high => hi,
                      :transform => x -> x,
                      :transform_inv => x -> x)
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
    x_min = cfg[:low]
    x_max = cfg[:high]
    if isnothing(x_min)
        x_min = 0.0
    end
    if isnothing(x_max)
        # Divide by 1.0 if no max value
        Δ = 1.0
    else
        Δ = x_max - x_min
    end
    if inv
        F_inv = cfg[:transform_inv]
        return x -> F_inv(x*Δ + x_min)
    else
        F = cfg[:transform]
        return x -> F((x - x_min)/Δ)
    end
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
        F_inv = opt_scaler_function(config, k, inv = true)

        rel_max = cfg[:rel_max]
        rel_min = cfg[:rel_min]
        abs_max = cfg[:abs_max]
        abs_min = cfg[:abs_min]
        for i in 1:n
            k = i + offset
            val = vals[i]
            if isnothing(rel_min)
                low = abs_min
            else
                low = max(abs_min, rel_min*val)
            end
            if isnothing(rel_max)
                hi = abs_max
            else
                hi  = min(abs_max, rel_max*val)
            end
            # We have found limits in terms of unscaled variable, scale on the way out
            low = F(low)
            hi = F(hi)
            @assert low <= x0[k]
            @assert hi >= x0[k]
            x_min[k] = low
            x_max[k] = hi
        end
    end
    return lims
end
