using Terv
using LinearAlgebra
using Printf
# using Plots
using ForwardDiff
# Turn on debugging to show output and timing.
# Turn on by uncommenting or running the following:
ENV["JULIA_DEBUG"] = Terv
# To disable the debug output:
# ENV["JULIA_DEBUG"] = nothing


function perform_test(nc = 100, tstep = repeat([0.02], 150))
    # Minimal TPFA grid: Simple grid that only contains connections and
    # fields required to compute two-point fluxes
    G = get_1d_reservoir(nc, z_max = 1)
    nc = number_of_cells(G)
    # Parameters
    bar = 1e5
    p0 = 100*bar # 100 bar
    mu = 1e-3    # 1 cP
    pRef = 100*bar
    # Liquid
    rhoLS = 1000
    cl = 1e-5/bar
    # Vapor
    rhoVS = 100
    cv = 1e-4/bar


    L = LiquidPhase()
    V = VaporPhase()
    sys = ImmiscibleSystem([L, V])
    model = SimulationModel(G, sys)
    s = model.secondary_variables
    s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, pRef, [rhoLS, rhoVS], [cl, cv])

    # Put heavy phase on top and light phase on bottom
    nl = nc ÷ 2
    sL = vcat(ones(nl), zeros(nc - nl))
    s0 = vcat(sL', 1 .- sL')

    init = Dict(:Pressure => p0, :Saturations => s0)
    state0 = setup_state(model, init)
    # Model parameters
    parameters = setup_parameters(model)
    parameters[:ReferenceDensity] = [rhoLS, rhoVS]
    timesteps = tstep*3600*24

    sim = Simulator(model, state0 = state0, parameters = parameters)
    cfg = simulator_config(sim, max_nonlinear_iterations = 20)
    println("Starting simulation.")
    states = simulate(sim, timesteps, config = cfg)
    s = states[end]
    p = s.Pressure
    @printf("Final pressure ranges from %f to %f bar.\n", maximum(p)/bar, minimum(p)/bar)
    sl = s.Saturations[1, :]
    @printf("Final liquid saturation ranges from %f to %f.\n", maximum(sl), minimum(sl))
    return states, model
end
## Perform test, with plotting
# states, model = perform_test(5000, repeat([0.01], 3000))
states, model = perform_test()
##
using Makie
tmp = vcat(map((x) -> x.Saturations[1, :]', states)...)
f = Figure()
ax = Axis(f[1, 1])
Makie.heatmap!(tmp')
ax.xlabel = "Depth"
ax.ylabel = "Time"
display(f)
