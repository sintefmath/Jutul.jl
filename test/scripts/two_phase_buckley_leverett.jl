using Jutul
function perform_test(;nc = 100, time = 1.0, nstep = 100)
    tstep = repeat([time/nstep], nstep)
    G = get_1d_reservoir(nc)
    nc = number_of_cells(G)
    timesteps = tstep*3600*24 # Convert time-steps from days to seconds

    # Definition of fluid phases
    bar = 1e5
    p0 = 100*bar
    L, V = LiquidPhase(), VaporPhase()
    # Define system and realize on grid
    sys = ImmiscibleSystem([L, V])
    model = SimulationModel(G, sys)
    s = model.secondary_variables
    s[:RelativePermeabilities] = BrooksCoreyRelPerm(sys, [2.0, 2.0], [0.2, 0.2])
    s[:PhaseViscosities] = ConstantVariables([1e-3, 5e-3]) # 1 and 5 cP

    tot_time = sum(timesteps)
    irate = 500*sum(G.grid.pore_volumes)/tot_time
    src  = [SourceTerm(1, irate, fractional_flow = [1.0, 0.0]), 
            SourceTerm(nc, -irate, fractional_flow = [1.0, 0.0])]
    forces = build_forces(model, sources = src)

    state0 = setup_state(model, Dict(:Pressure => p0, :Saturations => [0.0, 1.0]))
    # Simulate and return
    sim = Simulator(model, state0 = state0)
    states, report = simulate(sim, timesteps, forces = forces)
    return states, model, report
end
## Perform test
n, n_f = 100, 1000
states, model, report = perform_test(nc = n)
states_refined, = perform_test(nc = n_f, nstep = 1000)
## Plot results
using GLMakie
x = range(0, stop = 1, length = n)
x_f = range(0, stop = 1, length = n_f)
tmp = vcat(map((x) -> x.Saturations[1, :]', states)...)
f = Figure()
ax = Axis(f[1, 1], ylabel = "Saturation", title = "Buckley-Leverett")
for i in 1:6:length(states)
    GLMakie.lines!(ax, x, states[i].Saturations[1, :], color = :darkgray)
end
# Plot refined reference
GLMakie.lines!(ax, x_f, states_refined[end].Saturations[1, :], color = :red)
display(f)
