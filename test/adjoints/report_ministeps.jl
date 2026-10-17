using Jutul, Test

@testset "Adjoints require the forward ministeps" begin
    sys = VariablePoissonSystem(time_dependent = true)
    grid = CartesianMesh((3, 1), (1.0, 1.0))
    domain = DiscretizedDomain(grid, (poisson = Jutul.PoissonDiscretization(grid),))
    model = SimulationModel(domain, sys)
    state0 = setup_state(model; U = 1.0)
    parameters = setup_parameters(model; K = compute_face_trans(grid, 1.0))
    forces = setup_forces(model; sources = [PoissonSource(1, 1.0), PoissonSource(3, -1.0)])
    dt = [1.0, 1.0]
    case = JutulCase(model, dt, forces; state0, parameters)
    objective = (m, s, t, step, f) -> t * s[:U][end]^2
    function forward(params; substates = false)
        sim = Simulator(model; state0, parameters = params)
        cfg = simulator_config(
            sim; info_level = -1, output_substates = substates,
            timestep_selectors = [TimestepSelector(initial_absolute = 0.25, max = 0.25)]
        )
        return simulate(sim, dt; forces, config = cfg)
    end
    coarse = forward(parameters)
    @test length(report_timesteps(coarse.reports; ministeps = true)) == 8
    @test_throws ArgumentError solve_adjoint_sensitivities(model, coarse.states, coarse.reports, objective; state0, parameters, forces)
    @test_throws ArgumentError solve_adjoint_sensitivities(case, coarse, objective)
    @test_throws ArgumentError Jutul.AdjointPackedResult(coarse, forces)
    # Legacy optimization also consumes report states directly.
    @test_throws ArgumentError Jutul.gradient_opt!(zeros(1), zeros(1), Dict(:states => coarse.states, :reports => coarse.reports))
    fine = forward(parameters; substates = true)
    grad = solve_adjoint_sensitivities(case, fine, objective)
    ss, ts, ix = Jutul.expand_to_ministeps(fine)
    explicit = solve_adjoint_sensitivities(model, ss, ts, objective; state0, parameters, forces)
    @test grad[:K] ≈ explicit[:K]
    function value(params)
        s, t, ix = Jutul.expand_to_ministeps(forward(params; substates = true))
        return sum(objective(model, x, Δt, i, forces) for (i, (x, Δt)) in enumerate(zip(s, t)))
    end
    h = 1.0e-5 * parameters[:K][1]
    plus, minus = deepcopy(parameters), deepcopy(parameters)
    plus[:K][1] += h
    minus[:K][1] -= h
    numerical = (value(plus) - value(minus)) / (2h)
    @test isapprox(grad[:K][1], numerical; rtol = 1.0e-6)
    # A failed retry does not count as an additional successful ministep.
    reports = [Dict(:ministeps => [Dict(:success => false, :dt => 0.1), Dict(:success => true, :dt => t)]) for t in ts]
    single = solve_adjoint_sensitivities(model, ss, reports, objective; state0, parameters, forces)
    @test single[:K] ≈ explicit[:K]
end
