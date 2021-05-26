using Terv
using LinearAlgebra
using Printf
using Makie
using ForwardDiff
# Turn on debugging to show output and timing.
# Turn on by uncommenting or running the following:
ENV["JULIA_DEBUG"] = Terv
# To disable the debug output:
# ENV["JULIA_DEBUG"] = nothing
casename = "pico"
# casename = "2cell"
# casename = "spe10_symrmc"

function perform_test(casename, doPlot = false, pvfrac=0.05, tstep = [1.0, 2.0])
    # Minimal TPFA grid: Simple grid that only contains connections and
    # fields required to compute two-point fluxes
    G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true)
    nc = number_of_cells(G)
    # Parameters
    bar = 1e5
    p0 = 100*bar # 100 bar
    mu = 1e-3    # 1 cP
    cl = 1e-5/bar
    pRef = 100*bar
    rhoLS = 1000
    # Anonymous function for liquid density
    rhoL = (p) -> rhoLS*exp((p - pRef)*cl)
    # Single-phase liquid system (compressible pressure equation)
    L = LiquidPhase()
    V = VaporPhase()
    sys = ImmiscibleSystem([L, V])
    # Simulation model wraps grid and system together with context (which will be used for GPU etc)
    model = SimulationModel(G, sys)

    # System state
    pv = model.domain.grid.pore_volumes
    timesteps = tstep*3600*24 # 1 day, 2 days
    tot_time = sum(timesteps)
    irate = pvfrac*sum(pv)/tot_time
    src  = [SourceTerm(1, irate, fractional_flow = [1.0, 0.0]), 
            SourceTerm(nc, -irate)]
    forces = build_forces(model, sources = src)

    # State is dict with pressure in each cell
    init = Dict(:Pressure => p0, :Saturations => [0.0, 1.0])
    state0 = setup_state(model, init)
    # Model parameters
    parameters = setup_parameters(model)
    parameters[:Density] = [rhoL, rhoL]
    parameters[:CoreyExponents] = [2, 3]
    parameters[:Viscosity] = [mu, mu/2]

    sim = Simulator(model, state0 = state0, parameters = parameters)
    # Linear solver
    println("Starting simulation.")
    states = simulate(sim, timesteps, forces = forces)
    @show states
    s = states[end]
    p = s.Pressure
    @printf("Final pressure ranges from %f to %f bar.\n", maximum(p)/bar, minimum(p)/bar)
    sl = s.Saturations[1, :]
    @printf("Final liquid saturation ranges from %f to %f.\n", maximum(sl), minimum(sl))

    if doPlot
        ax = plot_interactive(mrst_data["G"], states)
        display(ax)
    else
        ax = nothing
    end
    return (sim, ax)
end
doPlot = false
sim, ax = perform_test(casename, doPlot)
ax
println("All done.")
