using Terv
using Test


function test_twophase(casename = "pico", pvfrac=0.05, tstep = [1.0, 2.0])
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
    L = LiquidPhase()
    V = VaporPhase()
    sys = ImmiscibleSystem([L, V])
    # Simulation model wraps grid and system together with context (which will be used for GPU etc)
    model = SimulationModel(G, sys)

    # System state
    timesteps = tstep*3600*24 # 1 day, 2 days
    tot_time = sum(timesteps)
    irate = pvfrac*sum(G.pv)/tot_time
    @show irate
    src = sources = [SourceTerm(1, irate, fractional_flow = [1.0, 0.0]), 
                     SourceTerm(nc, -irate)]
    # State is dict with pressure in each cell
    init = Dict("Pressure" => p0, "Saturations" => [0.0, 1.0])
    state0 = setup_state(model, init)
    # Model parameters
    parameters = setup_parameters(model)
    parameters["CoreyExponent_L"] = 2
    parameters["Density"] = [rhoL, rhoL]
    # Repeat same properties
    parameters["CoreyExponents"] = [2, 3]
    parameters["Viscosity"] = [mu, mu/2]

    sim = Simulator(model, state0 = state0, parameters = parameters)
    println("Starting simulation.")
    states = simulate(sim, timesteps, sources = src)
    return true
end

@test test_twophase()

