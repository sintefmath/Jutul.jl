using Jutul
using Test
using SparseArrays

function basic_poisson_case(nx = 3, ny = 1)
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
    case = JutulCase(model, dt, forces, state0 = state0, parameters = param)
    return case
end

function basic_poisson_test()
    case = basic_poisson_case()
    states, = simulate(case, info_level = -1)
    U = states[end][:U]
    # Singular problem, normalize against first element
    U = U .- U[1]
    @test U ≈ [0.0, 1/3, 2/3]
end

@testset "Variable Poisson" begin
    basic_poisson_test()
end

@testset "data_domain gradients" begin
    case = basic_poisson_case()
    model = case.model
    domain = model.data_domain
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
    result = simulate(case, info_level = -1)
    obj = (model, state, dt_n, n, forces_for_step_n) -> sum(state[:U])
    sens = solve_adjoint_sensitivities(case, result, obj)
    data_domain_with_gradients = Jutul.data_domain_to_parameters_gradient(model, sens)
    @test data_domain_with_gradients[:poisson_coefficient] ≈ [-0.33333492279052723, -0.4999980926513673, -0.1666631698608399] rtol=1e-3
    @test data_domain_with_gradients[:volumes] ≈ [0.0, 0.0, 0.0]
    @test data_domain_with_gradients[:areas] ≈ [-2/3, -1/3] rtol=1e-3
end
