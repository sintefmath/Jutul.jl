using Terv
using Test

function test_cc(linear_solver=nothing)
    state0, model, prm, f, t = get_test_setup_battery()
    sim = Simulator(model, state0=state0, parameters=prm)
    #! Have not change anything below
    cfg = simulator_config(sim)
    cfg[:linear_solver] = linear_solver
    simulate(sim, t, forces = f, config = cfg)
    # We just return true. The test at the moment just makes sure that the simulation runs.
    return true
end


test_cc()
# @testset "Single-phase" begin
#     @test test_cc()
# end

# @testset "Single-phase linear solvers" begin
#     @test test_cc(linear_solver = AMGSolver("RugeStuben", 1e-3))
#     @test test_cc(linear_solver = AMGSolver("SmoothedAggregation", 1e-3))
# end