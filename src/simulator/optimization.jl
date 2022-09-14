
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

function setup_parameter_optimization(model, state0, param, dt, forces, G, opt_cfg = optimization_config(model);
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
        dFdx .= grad_adj
        return dFdx
    end
    return (F, dF, x0, data)
end

function optimization_config(model, active = keys(model.parameters))
    out = Dict{Symbol, Any}()
    for k in active
        var = model.parameters[k]
        out[k] = Dict(:active => true,
                      :min => minimum_value(var),
                      :max => maximum_value(var),
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
    x_min = cfg[:min]
    x_max = cfg[:max]
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
