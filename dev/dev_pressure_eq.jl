using Terv
using LinearAlgebra
using Printf
using Makie
using ForwardDiff
# Turn on debugging to show output and timing.
# Turn on by uncommenting or running the following:
# ENV["JULIA_DEBUG"] = Terv
# To disable the debug output:
# ENV["JULIA_DEBUG"] = nothing
casename = "pico"
function perform_test(casename)
    # Minimal TPFA grid: Simple grid that only contains connections and
    # fields required to compute two-point fluxes
    G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true)

    println("Setting up simulation case.")
    nc = number_of_cells(G)
    nf = number_of_faces(G)

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
    phase = LiquidPhase()
    sys = SinglePhaseSystem(phase)
    # Simulation model wraps grid and system together with context (which will be used for GPU etc)
    model = SimulationModel(G, sys)

    # System state
    timesteps = [1.0, 2.0]*3600*24 # 1 day, 2 days
    tot_time = sum(timesteps)
    irate = sum(G.pv)/tot_time
    src = sources = [SourceTerm(1, [irate]), SourceTerm(nc, [-irate])]
    # State is dict with pressure in each cell
    state0 = setup_state(model, p0)
    # Model parameters
    parameters = setup_parameters(model)
    parameters[subscript("Viscosity", phase)] = mu
    parameters[subscript("Density", phase)] = rhoL

    sim = Simulator(model, state0 = state0, parameters = parameters)
    # Linear solver
    lsolve = AMGSolver()
    println("Starting simulation.")
    tol = 1e-6
    maxIt = 10

    states = simulate(sim, timesteps, sources = src, linsolve = lsolve)
    s = states[end]
    p = s["Pressure"]
    plot_mrstdata(mrst_data["G"], p/bar)

    # Uncomment to see final Jacobian
    # display(storage["LinearizedSystem"].jac)
end

perform_test(casename)