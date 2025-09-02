using Jutul
using Test

function test_heat_2d(nx = 3, ny = nx; kwarg...)
    sys = SimpleHeatSystem()
    # Unit square
    g = CartesianMesh((nx, ny), (1.0, 1.0))
    # Set up a model with the grid and system
    D = DiscretizedDomain(g)
    model = SimulationModel(D, sys)
    # Initial condition is random values
    nc = number_of_cells(g)
    T0 = rand(nc)
    state0 = setup_state(model, Dict(:T=>T0))
    sim = Simulator(model, state0 = state0)
    states, = simulate(sim, [1.0]; info_level = -1, kwarg...)
    return states
end

@testset "Simple heat equation 2D" begin
    @test begin
        states = test_heat_2d(4, 4)
        true
    end
end

using HYPRE
@testset "Algebraic multigrid heat" begin
    lsolve = GenericKrylov(:bicgstab, preconditioner = Jutul.AMGPreconditioner(:smoothed_aggregation))
    states = test_heat_2d(4, 4, linear_solver = lsolve)
    @test length(states) == 1

    lsolve = GenericKrylov(:bicgstab, preconditioner = Jutul.BoomerAMGPreconditioner())
    states = test_heat_2d(4, 4, linear_solver = lsolve)
    @test length(states) == 1
end
