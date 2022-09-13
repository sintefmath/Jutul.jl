
export update_objective_new_parameters!

function update_objective_new_parameters!(param_serialized, sim, state0, param, forces, dt, G; config = simulator_config(sim), kwarg...)
    devectorize_variables!(param, sim.model, param_serialized, :parameters)
    states, = simulate(state0, sim, dt, parameters = param, forces = forces; kwarg...)
    return evaluate_objective(G, sim.model, states, dt, forces)
end
