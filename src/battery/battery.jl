using Terv
export ElectroChemicalComponent, CurrentCollector
export get_test_setup_battery, get_cc_grid

abstract type ElectroChemicalComponent <: TervSystem end
struct CurrentCollector <: ElectroChemicalComponent end

# Instead of DarcyMassMobilityFlow
#? Is this the correct substitution
struct ChargeFlow <: FlowType end
include_face_sign(::ChargeFlow) = false

# abstract type PhaseAndComponentVariable <: GroupedVariables end
# #? Is this the right replacement for MassMobility
# abstract type PhaseMassDensities <: PhaseVariables end

struct Phi <: ScalarVariable end

function build_forces(model::SimulationModel{G, S}; sources = nothing) where {G<:TervDomain, S<:CurrentCollector}
    return (sources = sources,)
end

function get_test_setup_battery(context = "cpu", timesteps = [1.0, 2.0], pvfrac = 0.05)
    G = get_cc_grid()

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
    # ? Can i just skip this??
    # s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, pRef, rhoLS, cl)

    ## Bellow is not fixed

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


abstract type ElectroChemicalGrid <: TervGrid end

# TPFA grid
"Minimal struct for TPFA-like grid. Just connection data and pore-volumes"
struct MinimalECTPFAGrid{R<:AbstractFloat, I<:Integer} <: ElectroChemicalGrid
    pore_volumes::AbstractVector{R}
    neighborship::AbstractArray{I}
    function MinimalECTPFAGrid(pv, N)
        nc = length(pv)
        pv::AbstractVector
        @assert size(N, 1) == 2  "Two neighbors per face"
        if length(N) > 0
            @assert minimum(N) > 0   "Neighborship entries must be positive."
            @assert maximum(N) <= nc "Neighborship must be limited to number of cells."
        end
        @assert all(pv .> 0)     "Pore volumes must be positive"
        new{eltype(pv), eltype(N)}(pv, N)
    end
end

function degrees_of_freedom_per_unit(
    model::SimulationModel{D, S}, sf::ComponentVariable
    ) where {D<:TervDomain, S<:CurrentCollector}
    return 1 
end


function get_cc_grid(perm = nothing, poro = nothing, volumes = nothing, extraout = false)
    name = "pico"
    fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, ".mat")
    @debug "Reading MAT file $fn..."
    exported = MAT.matread(fn)
    @debug "File read complete. Unpacking data..."

    N = exported["G"]["faces"]["neighbors"]
    N = Int64.(N)
    internal_faces = (N[:, 2] .> 0) .& (N[:, 1] .> 0)
    N = copy(N[internal_faces, :]')
        
    # Cells
    cell_centroids = copy((exported["G"]["cells"]["centroids"])')
    # Faces
    face_centroids = copy((exported["G"]["faces"]["centroids"][internal_faces, :])')
    face_areas = vec(exported["G"]["faces"]["areas"][internal_faces])
    face_normals = exported["G"]["faces"]["normals"][internal_faces, :]./face_areas
    face_normals = copy(face_normals')
    if isnothing(perm)
        perm = copy((exported["rock"]["perm"])')
    end

    # Deal with cell data
    if isnothing(poro)
        poro = vec(exported["rock"]["poro"])
    end
    if isnothing(volumes)
        volumes = vec(exported["G"]["cells"]["volumes"])
    end
    pv = poro.*volumes

    @debug "Data unpack complete. Starting transmissibility calculations."
    # Deal with face data
    T_hf = compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, N)
    T = compute_face_trans(T_hf, N)

    G = MinimalECTPFAGrid(pv, N)
    if size(cell_centroids, 1) == 3
        z = cell_centroids[3, :]
        g = gravity_constant
    else
        z = nothing
        g = nothing
    end

    ft = ChargeFlow()
    flow = TwoPointPotentialFlow(SPU(), TPFA(), ft, G, T, z, g)
    disc = (mass_flow = flow,)
    D = DiscretizedDomain(G, disc)

    if extraout
        return (D, exported)
    else
        return D
    end
end


# To get right number of dof for CellNeigh...
function single_unique_potential(
    model::SimulationModel{D, S}
    )where {D<:TervDomain, S<:CurrentCollector}
    return false
end


function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type::ChargeFlow)
    S[:CellNeighborPotentialDifference] = CellNeighborPotentialDifference()
end

