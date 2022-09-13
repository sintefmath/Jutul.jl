
export update_objective_new_parameters!

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
    return obj
end
