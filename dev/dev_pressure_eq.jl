using Terv
using LinearAlgebra
using Printf
# Turn on debugging to show output and timing
ENV["JULIA_DEBUG"] = Terv
# ENV["JULIA_DEBUG"] = nothing
function perform_test(casename)
    # Minimal TPFA grid: Simple grid that only contains connections and
    # fields required to compute two-point fluxes
    G = get_minimal_tpfa_grid_from_mrst(casename)

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
    # Model with allocated storage. Will be moved into function at some point
    model = SimulationModel(G, sys)
    storage = allocate_storage(model)

    # System state
    p = repeat([p0], nc) # 100
    src = sources = [SourceTerm(1, [1.0]), SourceTerm(nc, [-1.0])]
    # State is dict with pressure in each cell
    state0 = Dict()
    state0["Pressure"] = p
    # Need two states for time-stepping - current step, previous step
    state = deepcopy(state0)
    # Assign in primary variable
    pAD = allocate_vector_ad(state0["Pressure"], 1, diag_pos = 1)
    state["Pressure"] = pAD

    storage["state0"] = state0
    storage["state"] = state
    # Model parameters
    parameters = Dict()
    parameters["Viscosity_L"] = mu
    parameters["Density_L"] = rhoL
    storage["parameters"] = parameters
    # Linear solver
    lsolve = AMGSolver()
    println("Starting simulation.")
    tol = 1e-6
    maxIt = 10
    for i = 1:maxIt
        e, tol = newton_step(model, storage, dt = 1, sources = src, iteration = i, linsolve = lsolve)
        if e < tol
            break
        end
    end
    println("Simulation complete.")
    # Uncomment to see final Jacobian
    # display(storage["LinearizedSystem"].jac)
end

perform_test("pico")