using Jutul
function perform_test(nc = 100, tstep = repeat([0.02], 150))
    G = get_1d_reservoir(nc, z_max = 1)
    nc = number_of_cells(G)
    # Definition of fluid phases
    bar = 1e5
    p0 = 100*bar
    rhoLS, rhoVS = 1000.0, 100.0
    cl, cv = 1e-5/bar, 1e-4/bar
    L, V = LiquidPhase(), VaporPhase()
    # Define system and realize on grid
    sys = ImmiscibleSystem([L, V])
    model = SimulationModel(G, sys)
    # Replace density with a lighter pair
    s = model.secondary_variables
    s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, p0, [rhoLS, rhoVS], [cl, cv])
    # Put heavy phase on top and light phase on bottom
    nl = nc ÷ 2
    sL = vcat(ones(nl), zeros(nc - nl))
    s0 = vcat(sL', 1 .- sL')
    state0 = setup_state(model, Dict(:Pressure => p0, :Saturations => s0))
    timesteps = tstep*3600*24 # Convert time-steps from days to seconds
    # Simulate and return
    sim = Simulator(model, state0 = state0)
    states, report = simulate(sim, timesteps)
    return states, model, report
end
## Perform test
states, model, report = perform_test();
## Plot results
using GLMakie
tmp = vcat(map((x) -> x[:Saturations][1, :]', states)...)
f = Figure()
ax = Axis(f[1, 1], xlabel = "Depth", ylabel = "Time", title = "Gravity segregation")
GLMakie.heatmap!(ax, tmp)
display(f)
