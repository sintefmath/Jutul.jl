using Jutul
using Test

function basic_poisson_test()
    nx = 3
    ny = 1
    sys = VariablePoissonSystem()
    # Unit square
    g = CartesianMesh((nx, ny), (1.0, 1.0))
    # Create domain with coefficient
    domain = DataDomain(g, poisson_coefficient = 1.0)
    # Set up a model with the grid and system
    model = SimulationModel(domain, sys)
    # Initial condition doesn't matter
    state0 = setup_state(model, Dict(:U=>1.0))
    # Set up parameters from domain data and model
    param = setup_parameters(model)

    nc = number_of_cells(g)
    pos_src = PoissonSource(1, 1.0)
    neg_src = PoissonSource(nc, -1.0)
    forces = setup_forces(model, sources = [pos_src, neg_src])

    dt = [1.0]
    states, = simulate(
        state0,
        model,
        dt,
        parameters = param,
        info_level = -1,
        forces = forces)
    U = states[end][:U]
    # Singular problem, normalize against first element
    U = U .- U[1]
    @test U ≈ [0.0, 1/3, 2/3]
end

@testset "Variable Poisson" begin
    basic_poisson_test()
end
