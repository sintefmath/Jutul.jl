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
@test test_heat_residual()
