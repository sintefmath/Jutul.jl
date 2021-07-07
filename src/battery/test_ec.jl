using Terv
using Test

ENV["JULIA_DEBUG"] = Terv;

function test_ec(linear_solver=nothing)
    state0, model, prm, f, t = get_test_setup_ec_component()
    sim = Simulator(model, state0=state0, parameters=prm)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = linear_solver
    states = simulate(sim, t, forces = f, config = cfg)
    return state0, states, model
end

test_ec()