using Terv
using Test

function test_single_phase(casename = "pico", pvfrac=0.05, tstep = [1.0, 2.0])
    # Minimal TPFA grid: Simple grid that only contains connections and
    # fields required to compute two-point fluxes
    G = get_minimal_tpfa_grid_from_mrst(casename)
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
    phase = LiquidPhase()
    sys = SinglePhaseSystem(phase)
    # Simulation model wraps grid and system together with context (which will be used for GPU etc)
    model = SimulationModel(G, sys)

    # System state
    timesteps = tstep*3600*24 # 1 day, 2 days
    tot_time = sum(timesteps)
    irate = pvfrac*sum(G.pv)/tot_time
    src = [SourceTerm(1, irate), 
           SourceTerm(nc, -irate)]
    # State is dict with pressure in each cell
    init = Dict("Pressure" => p0)
    state0 = setup_state(model, init)
    # Model parameters
    parameters = setup_parameters(model)
    parameters["Viscosity"] = [mu]
    parameters["Density"] = [rhoL]

    sim = Simulator(model, state0 = state0, parameters = parameters)
    # Linear solver
    lsolve = AMGSolver("RugeStuben", 1e-3)
    simulate(sim, timesteps, sources = src, linsolve = lsolve)
    # We just return true. The test at the moment just makes sure that the simulation runs.
    return true
end
@testset "Single-phase" begin
    @test test_single_phase()
end
