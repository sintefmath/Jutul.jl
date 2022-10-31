using Jutul, Test

function poisson_test_objective(model, state)
    U = state[:U]
    return 2*U[1] + 3*U[end]
end

function poisson_test_objective_vec(model, state)
    return [poisson_test_objective(model, state), poisson_test_objective(model, state)]
end

function mytest(; nx = 3, ny = 1, dt = [1.0, 2.0, π], in_place = false, extra_timing = false, scalar_obj = true)
    sys = VariablePoissonSystem(time_dependent = true)
    # Unit square
    g = CartesianMesh((nx, ny), (1.0, 1.0))
    # Set up a model with the grid and system
    discretization = (poisson = Jutul.PoissonDiscretization(g), )
    D = DiscretizedDomain(g, discretization)
    model = SimulationModel(D, sys)
    # Initial condition doesn't matter
    state0 = setup_state(model, Dict(:U=>1.0))
    K = compute_face_trans(g, 1.0)
    param = setup_parameters(model, K = K)

    nc = number_of_cells(g)
    pos_src = PoissonSource(1, 1.0)
    neg_src = PoissonSource(nc, -1.0)
    forces = setup_forces(model, sources = [pos_src, neg_src])

    sim = Simulator(model, state0 = state0, parameters = param)
    states, reports = simulate(sim, dt, info_level = -1, forces = forces);

    n_grad = length(K)
    if scalar_obj
        # Scalar mode - we test the gradient of the scalar objective against the numerical version
        # Define objective
        G = (model, state, dt, step_no, forces) -> poisson_test_objective(model, state)

        if in_place
            grad_adj = zeros(n_grad)
            solve_in_place!(grad_adj, model, state0, states, param, dt, G, forces)
        else
            grad_adj = solve_out_of_place(model, state0, states, param, reports, G, forces)
        end
        # Check against numerical gradient
        grad_num = Jutul.solve_numerical_sensitivities(model, states, reports, G, :K,
                                forces = forces, state0 = state0, parameters = param)
        @test isapprox(grad_num, grad_adj, atol = 1e-4)
    else
        # Test vector objective
        G = (model, state, dt, step_no, forces) -> poisson_test_objective_vec(model, state)
        n_obj = 2
        if in_place
            grad_adj = zeros(n_grad, n_obj)
            solve_in_place!(grad_adj, model, state0, states, param, dt, G, forces, n_objective = n_obj)
        else
            grad_adj = solve_out_of_place(model, state0, states, param, reports, G, forces, n_objective = n_obj)
        end
        # We know that the objective is the same entry repeated twice, do an internal self-check.
        @test all(grad_adj[:, 1] == grad_adj[:, 2])
    end
end


function solve_in_place!(grad_adj, model, state0, states, param, dt, G, forces; kwarg...)
    storage = setup_adjoint_storage(model; state0 = state0, parameters = param, kwarg...)
    grad_adj = solve_adjoint_sensitivities!(grad_adj, storage, states, state0, dt, G, forces = forces)
end

function solve_out_of_place(model, state0, states, param, reports, G, forces; kwarg...)
    grad_adj = solve_adjoint_sensitivities(model, states, reports, G, 
    forces = forces, state0 = state0, parameters = param, raw_output = true; kwarg...)
    return grad_adj
end

@testset "simple adjoint sensitivities" begin
    for scalar_obj in [true, false]
        for in_place in [false, true]
            # Test single step since it hits less of the code
            mytest(dt = [1.0], in_place = in_place, scalar_obj = scalar_obj)
            # Test with multiple time-steps
            mytest(in_place = in_place, scalar_obj = scalar_obj)
        end
    end
end
