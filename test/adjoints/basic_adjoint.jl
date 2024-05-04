using Jutul, Test

function poisson_test_objective(model, state)
    U = state[:U]
    return 2*U[1] + 3*U[end]
end

function poisson_test_objective_vec(model, state)
    return [poisson_test_objective(model, state), poisson_test_objective(model, state)]
end

function solve_adjoint_forward_test_system(dim, dt)
    sys = VariablePoissonSystem(time_dependent = true)
    # Unit square
    g = CartesianMesh(dim, (1.0, 1.0))
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
    return (model, state0, states, reports, param, forces)
end

function test_basic_adjoint(; nx = 3, ny = 1, dt = [1.0, 2.0, π], in_place = false, scalar_obj = true)
    model, state0, states, reports, param, forces = solve_adjoint_forward_test_system((nx, ny), dt)
    K = param[:K]
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

function test_optimization_gradient(; nx = 3, ny = 1, dt = [1.0, 2.0, π], use_scaling = true, use_log = false)
    model, state0, states, reports, param, forces = solve_adjoint_forward_test_system((nx, ny), dt)
    ϵ = 1e-6
    num_tol = 1e-4

    K = param[:K]
    G = (model, state, dt, step_no, forces) -> poisson_test_objective(model, state)

    cfg = optimization_config(model, param, use_scaling = use_scaling, rel_min = 0.1, rel_max = 10)
    if use_log
        cfg[:K][:scaler] = :log
    end
    tmp = setup_parameter_optimization(model, state0, param, dt, forces, G, cfg, param_obj = true, print = false);
    F_o, dF_o, F_and_dF, x0, lims, data = tmp
    # Evaluate gradient first to initialize
    F0 = F_o(x0)
    # This interface is only safe if F0 was called with x0 first.
    dF_initial = dF_o(similar(x0), x0)

    dF_num = similar(dF_initial)
    num_grad!(dF_num, x0, ϵ, F_o)

    # Check around initial point
    @test isapprox(dF_num, dF_initial, rtol = num_tol)
    # Perturb the data in a few different directions and verify
    # the gradients there too. Use the F_and_dF interface, that
    # computes gradients together with the objective
    for delta in [1.05, 0.85, 0.325, 1.55]
        x_mod = delta.*x0
        dF_mod = similar(dF_initial)
        F_and_dF(NaN, dF_mod, x_mod)
        num_grad!(dF_num, x_mod, ϵ, F_o)
        @test isapprox(dF_num, dF_mod, rtol = num_tol)
    end
end

function num_grad!(dF_num, x0, ϵ, F_o)
    F_initial = F_o(x0)
    for i in eachindex(dF_num)
        x = copy(x0)
        x[i] += ϵ
        F_perturbed = F_o(x)
        dF_num[i] = (F_perturbed - F_initial)/ϵ
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
    @testset "adjoint" begin
        for scalar_obj in [true, false]
            for in_place in [false, true]
                # Test single step since it hits less of the code
                test_basic_adjoint(dt = [1.0], in_place = in_place, scalar_obj = scalar_obj)
                # Test with multiple time-steps
                test_basic_adjoint(in_place = in_place, scalar_obj = scalar_obj)
            end
        end
    end
    @testset "optimization interface" begin
        for use_log in [true, false]
            for use_scaling in [true, false]
                test_optimization_gradient(use_scaling = use_scaling, use_log = use_log)
            end
        end
    end
end

