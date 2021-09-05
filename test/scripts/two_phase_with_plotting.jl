using Terv
using LinearAlgebra
using Printf
using GLMakie
using ForwardDiff
# Turn on debugging to show output and timing.
# Turn on by uncommenting or running the following:
ENV["JULIA_DEBUG"] = Terv
# To disable the debug output:
# ENV["JULIA_DEBUG"] = nothing


function perform_test(casename, doPlot = false, pvfrac=1, tstep = ones(25))
    # Minimal TPFA grid: Simple grid that only contains connections and
    # fields required to compute two-point fluxes
    G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true)
    G = get_minimal_tpfa_grid_from_mrst(casename)
    nc = number_of_cells(G)
    # Parameters
    bar = 1e5
    p0 = 100*bar # 100 bar
    mu = 1e-3    # 1 cP
    cl = 1e-5/bar
    pRef = 100*bar
    rhoLS = 1000.0

    # Two phase liquid-vapor system
    L = LiquidPhase()
    V = VaporPhase()
    sys = ImmiscibleSystem([L, V])
    model = SimulationModel(G, sys)
    s = model.secondary_variables
    # Same density for both phases
    s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, pRef, rhoLS, cl)
    s[:RelativePermeabilities] = BrooksCoreyRelPerm(sys, [2, 3])
    s[:PhaseViscosities] = ConstantVariables([mu, mu/2])

    # System state
    pv = model.domain.grid.pore_volumes
    timesteps = tstep*3600*24 # 1 day, 2 days
    tot_time = sum(timesteps)
    irate = rhoLS*pvfrac*sum(pv)/(tot_time)
    src  = [SourceTerm(1, irate, fractional_flow = [1.0, 0.0]), 
            SourceTerm(nc, -irate, fractional_flow = [1.0, 0.0])]
    forces = build_forces(model, sources = src)

    # State is dict with pressure in each cell
    init = Dict(:Pressure => p0, :Saturations => [0.0, 1.0])
    state0 = setup_state(model, init)
    # Model parameters
    parameters = setup_parameters(model)
    parameters[:reference_densities] = [rhoLS, rhoLS]

    sim = Simulator(model, state0 = state0, parameters = parameters)
    cfg = simulator_config(sim, max_nonlinear_iterations = 20)
    println("Starting simulation.")
    states, = simulate(sim, timesteps, forces = forces, config = cfg)
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
## Perform test, with plotting
casename = "pico"
doPlot = true
perform_test(casename, doPlot)
println("All done with $casename case!")
