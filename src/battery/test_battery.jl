using Terv
using Test

function test_cc(linear_solver=nothing)
    state0, model, prm, f, t = get_test_setup_battery()
    sim = Simulator(model, state0=state0, parameters=prm)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = linear_solver
    simulate(sim, t, forces = f, config = cfg)
    return true
end

test_cc()
