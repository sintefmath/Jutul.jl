using Terv
export ElectroChemicalComponent, CurrentCollector
export get_test_setup_battery, get_cc_grid

abstract type ElectroChemicalComponent <: TervSystem end
struct CurrentCollector <: ElectroChemicalComponent end

# Instead of DarcyMassMobilityFlow
#? Is this the correct substitution
struct ChargeFlow <: FlowType end
include_face_sign(::ChargeFlow) = false


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

function build_forces(model::SimulationModel{G, S}; sources = nothing) where {G<:TervDomain, S<:CurrentCollector}
    return (sources = sources,)
end

function declare_units(G::MinimalECTPFAGrid)
    c = (unit = Cells(), count = length(G.pore_volumes))  # Cells equal to number of pore volumes
    f = (unit = Faces(), count = size(G.neighborship, 2)) # Faces
    return [c, f]
end



# Instead of CellNeighborPotentialDifference (?)
struct Phi <: ScalarVariable 
end


@terv_secondary function update_as_secondary!(
    pot, tv::Phi, model, param, Phi
    ) #should resisitvity be added?
    mf = model.domain.discretizations.mass_flow
    conn_data = mf.conn_data
    context = model.context
    if mf.gravity
        update_cell_neighbor_potential_difference_gravity!(
            pot, conn_data, Phi, context, kernel_compatibility(context)
            )
    else
        update_cell_neighbor_potential_difference!(
            pot, conn_data, Phi, context, kernel_compatibility(context)
            )
    end
end

function update_cell_neighbor_potential_difference_gravity!(
    dpot, conn_data, p, rho, context, ::KernelDisallowed
    )
    Threads.@threads for i in eachindex(conn_data)
        c = conn_data[i]
        for phno = 1:size(rho, 1)
            rho_i = view(rho, phno, :)
            @inbounds dpot[phno, i] = half_face_two_point_kgradp_gravity(
                c.self, c.other, c.T, p, c.gdz, rho_i
                )
        end
    end
end

function update_cell_neighbor_potential_difference_gravity!(
    dpot, conn_data, p, rho, context, ::KernelAllowed
    )
    @kernel function kern(dpot, @Const(conn_data), @Const(p), @Const(rho))
        ph, i = @index(Global, NTuple)
        c = conn_data[i]
        rho_i = view(rho, ph, :)
        dpot[ph, i] = half_face_two_point_kgradp_gravity(
            c.self, c.other, c.T, p, c.gdz, rho_i
            )
    end
    begin
        d = size(dpot)

        kernel = kern(context.device, context.block_size, d)
        event_jac = kernel(dpot, conn_data, p, rho, ndrange = d)
        wait(event_jac)
    end
end


### PHYSICS
### MUST BE REWRITTEN; ARE IN THEIR ORIGINAL FORM
### REMOVE GRAVITY; PRESSURE -> PHI


@inline function half_face_two_point_kgradp_gravity(
    c_self::I, c_other::I, T, p::AbstractArray{R}, gΔz, ρ::AbstractArray{R}
    ) where {R<:Real, I<:Integer}
    return -T*two_point_potential_drop_half_face(c_self, c_other, p, gΔz, ρ)
end

@inline function two_point_potential_drop_half_face(
    c_self, c_other, p::AbstractVector, gΔz, ρ
    )
    return two_point_potential_drop(p[c_self], value(p[c_other]), gΔz, ρ[c_self], value(ρ[c_other]))
end

@inline function two_point_potential_drop(
    p_self::Real, p_other::Real, gΔz::Real, ρ_self::Real, ρ_other::Real
    )
    ρ_avg = 0.5*(ρ_self + ρ_other)
    return p_self - p_other + gΔz*ρ_avg
end

@inline function half_face_two_point_kgradp(
    c_self::I, c_other::I, T, p::AbstractArray{R}
    ) where {R<:Real, I<:Integer}
    return -T*(p[c_self] - value(p[c_other]))
end

@inline function half_face_two_point_flux_fused(
    c_self::I, c_other::I, T, p::AbstractArray{R}, λ::AbstractArray{R}, 
    gΔz, ρ::AbstractArray{R}
    ) where {R<:Real, I<:Integer}
    θ = half_face_two_point_kgradp_gravity(c_self, c_other, T, p, gΔz, ρ)
    λᶠ = spu_upwind(c_self, c_other, θ, λ)
    return λᶠ*θ
end

####




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
    rhoLS = 1000

    phi = 1

    sys = CurrentCollector()
    model = SimulationModel(G, sys, context = context)
    # ? Can i just skip this??
    # ? I think this is where resisitvity would go
    # s[:PhaseMassDensities] = ConstantCompressibilityDensities(sys, pRef, rhoLS, cl)


    # System state
    tot_time = sum(timesteps)
    irate = pvfrac*sum(pv)/tot_time
    src = [SourceTerm(1, irate), 
        SourceTerm(nc, -irate)]
    forces = build_forces(model, sources = src)

    # State is dict with pressure in each cell
    init = Dict(:Phi => phi)
    state0 = setup_state(model, init)
    # Model parameters
    parameters = setup_parameters(model)

    state0 = setup_state(model, init)
    return (state0, model, parameters, forces, timesteps)
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
    S[:Phi] = Phi()
end

