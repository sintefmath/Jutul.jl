using Jutul, Test
##
function test_heat_residual()
    nx = ny = 2
    sys = SimpleHeatSystem()
    g = CartesianMesh((nx, ny), (1.0, 1.0))
    D = DiscretizedDomain(g)
    model = SimulationModel(D, sys)
    nc = number_of_cells(g)
    T0 = rand(nc)
    state0 = setup_state(model, Dict(:T=>T0))
    sim = Simulator(model, state0 = state0)
    states, = simulate(sim, [1.0], info_level = -1, max_timestep_cuts = 0, max_nonlinear_iterations = 0)
    hsim = HelperSimulator(model, state0 = state0)
    x = vectorize_variables(model, state0)
    r = model_residual(hsim, x)
    r ≈ sim.storage.LinearizedSystem.r
end

function test_multimodel_residual()
    sys = ScalarTestSystem()
    A = B = ScalarTestDomain(use_manual = false)
    modelA = SimulationModel(A, sys)
    sourceA = ScalarTestForce(1.0)
    forcesA = setup_forces(modelA, sources = sourceA)
    state0A = setup_state(modelA, Dict(:XVar=>0.0))
    # Model B
    modelB = SimulationModel(B, sys)
    sourceB = ScalarTestForce(-1.0)
    forcesB = setup_forces(modelB, sources = sourceB)
    state0B = setup_state(modelB, Dict(:XVar=>0.0))

    model = MultiModel((A = modelA, B = modelB))
    add_cross_term!(model, ScalarTestCrossTerm(), target = :A, source = :B, equation = :test_equation)
    # Set up joint state and forces
    state0 = setup_state(model, A = state0A, B = state0B)
    forces = setup_forces(model, A = forcesA, B = forcesB)
    # Set up simulator, and run simulation
    sim = Simulator(model, state0 = state0)
    states, = simulate(sim, [1.0], info_level = -1, max_timestep_cuts = 0, max_nonlinear_iterations = 0)

    hsim = HelperSimulator(model, state0 = state0)
    x = vectorize_variables(model, state0)
    r = model_residual(hsim, x)
    r ≈ sim.storage.LinearizedSystem.r
end
@testset "Helper" begin
    @testset "Single model" begin
        @test test_heat_residual()
    end
    @testset "Multi model" begin
        @test test_multimodel_residual()
    end
end
