using Terv
using Test

ENV["JULIA_DEBUG"] = Terv;

##
function test_cc(linear_solver=nothing)
    state0, model, prm, f, t, G = get_test_setup_cc()
    sim = Simulator(model, state0=state0, parameters=prm)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = linear_solver
    states = simulate(sim, t, forces = f, config = cfg)
    return state0, states, model, G
end

state0, states, model, G = test_cc();
##
f = plot_interactive(G, states)
display(f)

###

states, G = test_mixed_boundary_conditions();
##
f = plot_interactive(G, states)
display(f)
