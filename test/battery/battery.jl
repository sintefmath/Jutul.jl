using Terv
export ElectroChemicalComponent, CurrentCollector
export get_test_setup_battery

abstract type ElectroChemicalComponent <: TervSystem end
struct CurrentCollector <: ElectroChemicalComponent end

function get_test_setup_battery(grid_name; context = "cpu", timesteps = [1.0, 2.0], pvfrac = 0.05)
    G = get_minimal_tpfa_grid_from_mrst(grid_name)

    nc = number_of_cells(G)
    pv = G.grid.pore_volumes
    timesteps = timesteps*3600*24

    if context == "cpu"
        context = DefaultContext()
    elseif isa(context, String)
        error("Unsupported target $context")
    end
    @assert isa(context, TervContext)

    #TODO: Gjøre til batteri-parametere
    # Parameters
    bar = 1e5
    p0 = 100*bar # 100 bar
    cl = 1e-5/bar
    pRef = 100*bar
    rhoLS = 1000

    sys = CurrentCollector()
    model = SimulationModel(G, sys, context = context)
    s = model.secondary_variables

    ## Bellow is not fixed
    s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, pRef, rhoLS, cl)

    # System state
    tot_time = sum(timesteps)
    irate = pvfrac*sum(pv)/tot_time
    src = [SourceTerm(1, irate), 
        SourceTerm(nc, -irate)]
    forces = build_forces(model, sources = src)

    # State is dict with pressure in each cell
    init = Dict(:Pressure => p0)
    state0 = setup_state(model, init)
    # Model parameters
    parameters = setup_parameters(model)

    state0 = setup_state(model, init)
    return (state0, model, parameters, forces, timesteps)
end