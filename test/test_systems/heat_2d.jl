using Jutul
using Test

function test_poisson(n = 10)
    sys = SimpleHeatSystem()
    # Unit square
    g = CartesianMesh((10, 10), (1.0, 1.0))
    D = DiscretizedDomain(g)
    model = SimulationModel(D, sys)
    
    nc = number_of_cells(g)
    T0 = rand(nc)
    state0 = setup_state(model, Dict(:T=>T0))
    sim = Simulator(model, state0 = state0)
    states, = simulate(sim, [1.0])

    T = states[end][:T]
    return T
end

@testset "Poisson 2D" begin
    @test test_poisson()
end
