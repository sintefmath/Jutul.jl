using Jutul
using Test

function test_poisson(nx = 3, ny = nx)
    sys = SimpleHeatSystem()
    # Unit square
    g = CartesianMesh((nx, ny), (1.0, 1.0))
    D = DiscretizedDomain(g)
    model = SimulationModel(D, sys)
    nc = number_of_cells(g)
    T0 = rand(nc)
    state0 = setup_state(model, Dict(:T=>T0))
    sim = Simulator(model, state0 = state0)
    states, = simulate(sim, [1.0], info_level = 3)
    return states
end

@testset "Poisson 2D" begin
    @test begin
        states = test_poisson(4, 4)
        true
    end
end
