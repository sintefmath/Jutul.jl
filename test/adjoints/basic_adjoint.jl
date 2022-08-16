using Jutul, Test

function poisson_test_objective(model, state)
    U = state[:U]
    return 2*U[1] + 3*U[end]
end


function mytest(; nx = 3, ny = 1, dt = [1.0, 2.0, π], extra_timing = false)
    sys = VariablePoissonSystem(time_dependent = true)
    # Unit square
    g = CartesianMesh((nx, ny), (1.0, 1.0))
    # Set up a model with the grid and system
    discretization = (poisson = Jutul.PoissonDiscretization(g), )
    D = DiscretizedDomain(g, discretization)
    model = SimulationModel(D, sys)
    # Initial condition doesn't matter
    state0 = setup_state(model, Dict(:U=>1.0))
    param = setup_parameters(model, K = compute_face_trans(g, 1.0))

    nc = number_of_cells(g)
    pos_src = PoissonSource(1, 1.0)
    neg_src = PoissonSource(nc, -1.0)
    forces = setup_forces(model, sources = [pos_src, neg_src])

    sim = Simulator(model, state0 = state0, parameters = param)
    states, reports = simulate(sim, dt, info_level = -1, forces = forces);

    # Define objective
    G = (model, state, dt, step_no, forces) -> poisson_test_objective(model, state)
    grad_adj = Jutul.solve_adjoint_sensitivities(model, states, reports, G,
                        forces = forces, state0 = state0, parameters = param, extra_timing = extra_timing)
    # Check against numerical gradient
    grad_num = Jutul.solve_numerical_sensitivities(model, states, reports, G, :K,
                            forces = forces, state0 = state0, parameters = param)
    @test isapprox(grad_num, grad_adj, atol = 1e-4)
end

@testset "simple adjoint sensitivities" begin
    # Test single step since it hits less of the code
    mytest(dt = [1.0])
    # Test with multiple time-steps
    mytest()
end
