using Terv
using Test

function test_single_phase(grid = "pico"; linear_solver = nothing, kwarg...)
    state0, model, prm, f, t = get_test_setup(grid, case_name = "single_phase_simple"; kwarg...)
    sim = Simulator(model, state0 = state0, parameters = prm)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = linear_solver
    simulate(sim, t, forces = f, config = cfg)
    # We just return true. The test at the moment just makes sure that the simulation runs.
    return true
end
@testset "Single-phase" begin
    @test test_single_phase()
    @test test_single_phase(fuse_flux = true)
end

@testset "Single-phase linear solvers" begin
    @test test_single_phase(linear_solver = AMGSolver(:ruge_stuben))
    @test test_single_phase(linear_solver = AMGSolver(:smoothed_aggregation, relative_tolerance = 1e-3))
    @test test_single_phase(linear_solver = GenericKrylov())
    @test test_single_phase(linear_solver = GenericKrylov(preconditioner = ILUZeroPreconditioner()))
end
