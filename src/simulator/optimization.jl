
export update_objective_new_parameters!, setup_parameter_optimization

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

function setup_parameter_optimization(model, state0, param, dt, forces, G; grad_type = :adjoint, config = nothing, kwarg...)
    x0 = vectorize_variables(model, param, :parameters)
    data = Dict()
    data[:n_objective] = 1
    data[:n_gradient] = 1

    sim = Simulator(model, state0 = state0, parameters = param)
    if isnothing(config)
        config = simulator_config(sim; info_level = -1, kwarg...)
    end

    function F(x)
        devectorize_variables!(param, sim.model, x, :parameters)
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
        devectorize_variables!(param, model, x, :parameters)
        storage = setup_adjoint_storage(model, state0 = state0, parameters = param)
        grad_adj = solve_adjoint_sensitivities!(grad_adj, storage, data[:states], state0, dt, G, forces = forces)
        tmp = store_sensitivities(storage.parameter.model, grad_adj)
        data[:n_gradient] += 1
        dFdx .= grad_adj
        return dFdx
    end
    return (F, dF, x0, data)
end
