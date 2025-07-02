using Jutul, Test

function poisson_test_objective(model, state)
    U = state[:U]
    return 2.5*(U[end] - U[1])
end

function poisson_test_objective_vec(model, state)
    return [poisson_test_objective(model, state), poisson_test_objective(model, state)]
end

function setup_poisson_test_case(dx, dy, U0, k_val, srcval; dim = (2, 2), dt = [1.0])
    sys = VariablePoissonSystem(time_dependent = true)
    # Unit square
    g = CartesianMesh(dim, (dx, dy))
    # Set up a model with the grid and system
    discretization = (poisson = Jutul.PoissonDiscretization(g), )
    D = DiscretizedDomain(g, discretization)
    model = SimulationModel(D, sys)
    state0 = setup_state(model, Dict(:U=>U0))
    K = compute_face_trans(g, k_val)
    param = setup_parameters(model, K = K)

    nc = number_of_cells(g)
    pos_src = PoissonSource(1, srcval)
    neg_src = PoissonSource(nc, -srcval)
    forces = setup_forces(model, sources = [pos_src, neg_src])
    return JutulCase(model, dt, forces; parameters = param, state0 = state0)
end

function solve_adjoint_forward_test_system(dim, dt)
    case = setup_poisson_test_case(1.0, 1.0, 1.0, 1.0, 1.0, dim = dim, dt = dt)
    (; state0, forces, model, parameters, dt) = case
    states, reports = simulate(state0, model, parameters = parameters, dt, info_level = -1, forces = forces);
    return (model, state0, states, reports, parameters, forces)
end
##
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
##
import Jutul.AdjointsDI: solve_adjoint_generic

function setup_poisson_test_case_from_vector(x::Vector; fmt = :case, kwarg...)
    case = setup_poisson_test_case(x...; kwarg...)
    # Various formats that the sensitivity code can use.
    if fmt == :case
        out = case
    elseif fmt == :onecase
        out = case[1:1]
    elseif fmt == :model
        # TODO: This part currently isn't tested.
        out = case.model
    elseif fmt == :model_and_prm
        out = (case.model, case.parameters)
    elseif fmt == :model_and_prm_and_forces
        out = (case.model, case.parameters, case.forces)
    elseif fmt == :model_and_prm_and_forces_and_state0
        out = (case.model, case.parameters, case.forces, case.state0)
    else
        error("Unknown format $fmt for setup_poisson_test_case_from_vector")
    end
    return out
end

function num_grad_generic(F, G, x0)
    out = similar(x0)
    ϵ = 1e-12
    function objective_from_x(xi)
        case = F(xi, missing)
        r = simulate(case, info_level = -1)
        return Jutul.evaluate_objective(G, case, r)
    end
    G0 = objective_from_x(x0)
    for i in eachindex(x0)
        x = copy(x0)
        x[i] += ϵ
        Gi = objective_from_x(x)
        out[i] = (Gi - G0)/ϵ
    end
    return out
end

function test_for_timesteps(timesteps; atol = 5e-3, fmt = :case, kwarg...)
    # dx, dy, U0, k_val, srcval
    x = ones(5)
    case = setup_poisson_test_case_from_vector(x, dt = timesteps)
    states, reports = simulate(case, info_level = -1)

    F = (x, step_info) -> setup_poisson_test_case_from_vector(x, dt = timesteps, fmt = fmt)
    F_num = (x, step_info) -> setup_poisson_test_case_from_vector(x, dt = timesteps, fmt = :case)
    G = (model, state, dt, step_info, forces) -> poisson_test_objective(model, state)
    dGdx_num = num_grad_generic(F_num, G, x)
    dGdx_adj = solve_adjoint_generic(x, F, states, reports, G;
        state0 = case.state0,
        forces = case.forces,
        kwarg...
    )

    if fmt == :model_and_prm || fmt == :model
        dGdx_adj = dGdx_adj[1:4]
        dGdx_num = dGdx_num[1:4]
    end
    @test dGdx_adj ≈ dGdx_num atol = atol
end

@testset "AdjointDI.solve_adjoint_generic" begin
    test_for_timesteps([1.0])
    # Sparse forwarddiff with sparsity recomputed
    test_for_timesteps([1.0], do_prep = false)
    # Non-sparse forwarddiff
    test_for_timesteps([1.0], backend = Jutul.AdjointsDI.AutoForwardDiff(), do_prep = false)

    test_for_timesteps([100.0])
    test_for_timesteps([10.0, 3.0, 500.0, 100.0], atol = 0.01)
    for fmt in [:case, :onecase, :model_and_prm, :model_and_prm_and_forces, :model_and_prm_and_forces_and_state0]
        test_for_timesteps([100.0], fmt = fmt)
    end
end

import Jutul.DictOptimization as DictOptimization
@testset "DictOptimization" begin
    testdata = Dict(
        "scalar" => 3.0,
        "nested" => Dict(
            "vector" => [5.0, 1.0, 2.0, 3.0],
            "scalar" => 3.0
        ),
        "negative_scalar" => -3.0,
        "vector" => [3.0, -1.0],
        "matrix" => [1.0 -pi; 5.0 4.0]
    )

    dopt = DictParameters(testdata)

    @test_throws "[\"scalar\"] has limit abs_min larger than initial value 3.0" DictOptimization.free_optimization_parameter!(dopt, "scalar", abs_min = 5.0, abs_max = 4.0)
    @test_throws "[\"scalar\"] has no feasible values for abs_min = 3.0 and abs_max = 3.0" DictOptimization.free_optimization_parameter!(dopt, "scalar", abs_min = 3.0, abs_max = 3.0)
    @test_throws "[\"vector\"] has limit abs_min larger than initial value -1.0 in entry at CartesianIndex(2,)." DictOptimization.free_optimization_parameter!(dopt, "vector", abs_min = 0.0, abs_max = 4.0)

    free_optimization_parameter!(dopt, "scalar", abs_min = -2.0, abs_max = 4.0)
    s = dopt.parameter_targets[["scalar"]]
    @test s.abs_min == -2.0
    @test s.abs_max == 4.0
    @test s.rel_min == -Inf
    @test s.rel_max == Inf

    freeze_optimization_parameter!(dopt, "scalar")
    @test !haskey(dopt.parameter_targets, ["scalar"])

    free_optimization_parameter!(dopt, "vector", abs_min = [0.2, -2.0], abs_max = 4.0)
    v = dopt.parameter_targets[["vector"]]
    @test v.abs_min == [0.2, -2.0]
    @test v.abs_max == 4.0
    @test v.rel_min == -Inf
    @test v.rel_max == Inf


    lims = DictOptimization.realize_limits(dopt, "vector")
    @test lims.min ≈ [0.2, -2.0]
    @test lims.max ≈ [4.0, 4.0]

    l = DictOptimization.KeyLimits(rel_min = 0.1, rel_max = 1.5)

    # Relative limits
    @test DictOptimization.realize_limit(1.0, l, is_max = true) ≈ 1.5
    @test DictOptimization.realize_limit(100.0, l, is_max = true) ≈ 150.0

    @test DictOptimization.realize_limit(1.0, l, is_max = false) ≈ 0.1
    @test DictOptimization.realize_limit(100.0, l, is_max = false) ≈ 10.0

    # Absolute limits
    l = DictOptimization.KeyLimits(abs_min = 0.1, abs_max = 150.0)
    @test DictOptimization.realize_limit(1.0, l, is_max = true) ≈ 150.0
    @test DictOptimization.realize_limit(100.0, l, is_max = true) ≈ 150.0

    @test DictOptimization.realize_limit(1.0, l, is_max = false) ≈ 0.1
    @test DictOptimization.realize_limit(100.0, l, is_max = false) ≈ 0.1

    # Mixed limits
    l = DictOptimization.KeyLimits(abs_min = 0.1, abs_max = 150.0, rel_min = 0.1, rel_max = 1.5)
    @test DictOptimization.realize_limit(1.0, l, is_max = true) ≈ 1.5
    @test DictOptimization.realize_limit(100.0, l, is_max = true) ≈ 150.0

    @test DictOptimization.realize_limit_inner(-1.0, Inf, 4.0, is_max = true) ≈ 4.0
    @test DictOptimization.realize_limit_inner(-1.0, 2.0, 4.0, is_max = true) ≈ 0.0

    @testset "optimizer" begin
        function default_poisson_dict()
            return Dict(
                "dx" => 1.0,
                "dy" => 1.0,
                "U0" => 1.0,
                "k_val" => 1.0,
                "srcval" => 1.0
            )
        end

        function setup_poisson_test_case_from_dict(d::AbstractDict, step_info = missing; fmt = :case, kwarg...)
            return setup_poisson_test_case(d["dx"], d["dy"], d["U0"], d["k_val"], d["srcval"]; dim = (2, 2), dt = [1.0])
        end

        prm_truth = default_poisson_dict()
        states, = simulate(setup_poisson_test_case_from_dict(prm_truth), info_level = -1)
        function poisson_mismatch_objective(m, s, dt, step_info, forces)
            step = step_info[:step]
            U = s[:U]
            U_ref = states[step][:U]
            v = sum(i -> (U[i] - U_ref[i]).^2, eachindex(U, U_ref))
            return dt*v
        end
        # Perturb a parameter
        prm = default_poisson_dict()
        prm["k_val"] = 3.333

        dprm = DictParameters(prm, setup_poisson_test_case_from_dict, verbose = false)
        free_optimization_parameter!(dprm, "k_val", abs_max = 10.0, abs_min = 0.1)
        # Also do one with relative limits that should not change much
        free_optimization_parameter!(dprm, "U0", rel_max = 10.0, rel_min = 0.1)

        prm_opt = optimize(dprm, poisson_mismatch_objective, max_it = 25);

        @test prm_opt["k_val"] ≈ prm_truth["k_val"] atol = 0.01
        @test prm_opt["U0"] ≈ prm_truth["U0"] atol = 0.01

        grad = parameters_gradient(dprm, poisson_mismatch_objective, setup_poisson_test_case_from_dict)
        @test grad["k_val"] ≈ 0.0276189 atol = 0.01
        @test grad["U0"] ≈ 0.00 atol = 1e-8
        @test !haskey(grad, "dx")
        @test !haskey(grad, "dy")
        @test !haskey(grad, "srcval")

        dprm.strict = false
        free_optimization_parameters!(dprm)
        # Test the version without explicitly passing the setup function
        grad_all = parameters_gradient(dprm, poisson_mismatch_objective)
        @test grad_all["k_val"] ≈ 0.0276189 atol = 0.01
        @test grad_all["U0"] ≈ 0.00 atol = 1e-8
        @test grad_all["dx"] ≈ 0.0 atol = 1e-8
        @test grad_all["dy"] ≈ 0.0 atol = 1e-8
        @test grad_all["srcval"] ≈ -0.105863 atol = 0.01
    end
end

##
@testset "AdjointPackedResult" begin
    dt_test = [1.0, 2.0, 3.0]
    case = setup_poisson_test_case(1.0, 1.0, 1.0, 1.0, 1.0, dt = dt_test)
    r = simulate(case, info_level = -1, output_substates = true, max_timestep = 0.5)
    pr = Jutul.AdjointPackedResult(r, case)
    dt_i = report_timesteps(r.reports, ministeps = true)
    @test length(pr) == length(dt_i)

    r2 = simulate(case, info_level = -1, output_substates = false, max_timestep = 0.5)
    pr2 = Jutul.AdjointPackedResult(r2, case)
    @test length(pr2) == length(case.dt)

    pr3 = Jutul.AdjointPackedResult(r2, missing)
    @test length(pr3) == length(case.dt)
    @test ismissing(pr3.forces)
    @test ismissing(pr3[2].forces)
end
