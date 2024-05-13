using Jutul
using Test
using SparseArrays

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
    state0 = setup_state(model, U = 1.0)
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

@testset "data_domain gradients" begin  
    nx = 3
    ny = 1
    sys = VariablePoissonSystem()
    g = CartesianMesh((nx, ny), (1.0, 1.0))
    domain = DataDomain(g, poisson_coefficient = 1.0)
    model = SimulationModel(domain, sys)
    for T in [Float64, Float32]
        x = Jutul.vectorize_data_domain(domain)
        x = T.(x)
        dnew = Jutul.devectorize_data_domain(domain, x)
        @testset "$T" begin
            for (k, val_e_pair) in pairs(domain.data)
                v0, e0 = val_e_pair
                v, e = dnew.data[k]
                @testset "$k" begin
                    @test size(v) == size(v0)
                    @test v ≈ v0
                    @test e == e0
                end
            end
        end
    end
    prm_ad = Jutul.data_domain_to_parameters_gradient(model)
    for (k, v) in pairs(prm_ad[:K])
        @test length(v) == number_of_faces(g)
    end
    @test prm_ad[:K][:areas][2] ≈ [0.0, 3.0]
    @test prm_ad[:K][:areas][1] ≈ [3.0, 0.0]
    @test all(all(iszero, prm_ad[:K][:boundary_centroids]))
end
