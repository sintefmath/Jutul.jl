using Terv
using Test

ENV["JULIA_DEBUG"] = Terv;

function test_ec(linear_solver=nothing)
    state0, model, prm, f, t, G = get_test_setup_ec_component()
    sim = Simulator(model, state0=state0, parameters=prm)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = linear_solver
    states = simulate(sim, t, forces = f, config = cfg)
    return states, G
end

states, G = test_ec();
f = plot_interactive(G, states)
display(f)
