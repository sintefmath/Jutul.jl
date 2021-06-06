# export half_face_flux, half_face_flux!, tp_flux, half_face_flux_kernel
export SPU, TPFA, TwoPointPotentialFlow, DarcyMassMobilityFlow, CellNeighborPotentialDifference

abstract type TwoPointDiscretization <: TervDiscretization end

abstract type PotentialFlowDiscretization <: TervDiscretization end
abstract type KGradDiscretization <: PotentialFlowDiscretization end

abstract type UpwindDiscretization <: TervDiscretization end

abstract type FlowType <: TervDiscretization end

function select_primary_variables_flow_type(S, domain, system, formulation, flow_type)

end

struct DarcyMassMobilityFlow <: FlowType end
function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type::DarcyMassMobilityFlow)
    S[:CellNeighborPotentialDifference] = CellNeighborPotentialDifference()
    S[:PhaseMobilities] = PhaseMobilities()
    S[:MassMobilities] = MassMobilities()
end

struct DarcyMassMobilityFlowFused <: FlowType end
function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type::DarcyMassMobilityFlowFused)
    S[:PhaseMobilities] = PhaseMobilities()
    S[:MassMobilities] = MassMobilities()
end

struct TotalMassVelocityMassFractionsFlow <: FlowType end
function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type::TotalMassVelocityMassFractionsFlow)
    S[:TotalMass] = TotalMass()
end

"""
Two-point flux approximation.
"""
struct TPFA <: KGradDiscretization end

"""
Single-point upwinding.
"""
struct SPU <: UpwindDiscretization end

"Discretization of kgradp + upwind"
abstract type FlowDiscretization <: TervDiscretization end

function get_connection(face, cell, faces, N, T, z, g)
    D = Dict()
    if N[1, face] == cell
        other = N[2, face]
    else
        other = N[1, face]
    end
    D[:self] = cell
    D[:other] = other
    D[:face] = face
    if !isnothing(T)
        D[:T] = T[face]
    end
    if !isnothing(z)
        D[:gdz] = g*(z[cell] - z[other])
    end
    return convert_to_immutable_storage(D)
end

struct TwoPointPotentialFlow{U <:UpwindDiscretization, K <:PotentialFlowDiscretization, F <: FlowType} <: FlowDiscretization
    upwind::U
    grad::K
    flow_type::F
    gravity::Bool
    conn_pos
    conn_data
end

function TwoPointPotentialFlow(u, k, flow_type, grid, T = nothing, z = nothing, gravity = gravity_constant)
    N = grid.neighborship
    faces, face_pos = get_facepos(N)
    has_grav = !isnothing(gravity) || gravity == 0

    nhf = length(faces)
    nc = length(face_pos) - 1
    if isnothing(z)
        if has_grav
            @warn "No depths (z) provided, but gravity is enabled."
        end
    else
        @assert length(z) == nc
    end
    if !isnothing(T)
        @assert length(T) == nhf ÷ 2
    end
    get_el = (face, cell) -> get_connection(face, cell, faces, N, T, z, gravity)
    el = get_el(1, 1) # Could be junk, we just need eltype
    
    conn_data = Vector{typeof(el)}(undef, nhf)
    Threads.@threads for cell = 1:nc
        @inbounds for fpos = face_pos[cell]:(face_pos[cell+1]-1)
            conn_data[fpos] = get_el(faces[fpos], cell)
        end
    end
    TwoPointPotentialFlow{typeof(u), typeof(k), typeof(flow_type)}(u, k, flow_type, has_grav, face_pos, conn_data)
end


function select_secondary_variables_discretization!(S, domain, system, formulation, fd::TwoPointPotentialFlow)
    select_secondary_variables_flow_type!(S, domain, system, formulation, fd.flow_type)
end

function transfer(context::SingleCUDAContext, fd::TwoPointPotentialFlow{U, K}) where {U, K}
    tf = (x) -> transfer(context, x)
    u = tf(fd.upwind)
    k = tf(fd.grad)
    conn_pos = tf(fd.conn_pos)
    conn_data = tf(fd.conn_data)

    return TwoPointPotentialFlow{U, K}(u, k, conn_pos, conn_data)
end

struct CellNeighborPotentialDifference <: GroupedVariables end

function single_unique_potential(model)
    # We should add capillary pressure here ventually
    return model.domain.discretizations.mass_flow.gravity
end

function degrees_of_freedom_per_unit(model, sf::CellNeighborPotentialDifference)
    if single_unique_potential(model)
        n = number_of_phases(model.system)
    else
        n = 1
    end
    return n
end

function associated_unit(::CellNeighborPotentialDifference)
    Cells()
end

function number_of_units(model, pv::CellNeighborPotentialDifference)
    # We have two units of potential difference per face of the domain since the difference
    # is taken with respect to cells
    return 2*count_units(model.domain, Faces())
end

@terv_secondary function update_as_secondary!(pot, tv::CellNeighborPotentialDifference, model, param, Pressure, PhaseMassDensities)
    mf = model.domain.discretizations.mass_flow
    conn_data = mf.conn_data
    context = model.context
    if mf.gravity
        update_cell_neighbor_potential_difference_gravity!(pot, conn_data, Pressure, PhaseMassDensities, context, kernel_compatibility(context))
    else
        update_cell_neighbor_potential_difference!(pot, conn_data, Pressure, context, kernel_compatibility(context))
    end
end

function update_half_face_flux!(law, storage, model, flowd::TwoPointPotentialFlow{U, K, T}) where {U,K,T<:DarcyMassMobilityFlow}
    @assert false
    p = storage.state.Pressure
    mmob = storage.state.MassMobilities

    flux = get_entries(law.half_face_flux_cells)
    conn_data = law.flow_discretization
    # half_face_flux!(flux, model, conn_data, mmob, p)
end


function update_cell_neighbor_potential_difference_gravity!(dpot, conn_data, p, rho, context, ::KernelDisallowed)
    Threads.@threads for i in eachindex(conn_data)
        c = conn_data[i]
        for phno = 1:size(rho, 1)
            rho_i = view(rho, phno, :)
            @inbounds dpot[phno, i] = half_face_two_point_kgradp_gravity(c.self, c.other, c.T, p, c.gdz, rho_i)
        end
    end
end

function update_cell_neighbor_potential_difference!(dpot, conn_data, p, context, ::KernelDisallowed)
    Threads.@threads for i in eachindex(conn_data)
        c = conn_data[i]
        @inbounds dpot[i] = half_face_two_point_kgradp(c.self, c.other, c.T, p)
    end
end

# Flux primitive functions follow
@inline function spu_upwind(c_self::I, c_other::I, θ, λ::AbstractArray{R}, p::AbstractArray{R}) where {R<:Real, I<:Integer}
    if θ < 0
        # Flux is leaving the cell
        λᶠ = λ[c_self]
    else
        # Flux is entering the cell
        λᶠ = value(λ[c_other])
    end
    return λᶠ*θ
end

@inline function half_face_two_point_kgradp_gravity(c_self::I, c_other::I, T, p::AbstractArray{R}, gΔz, ρ::AbstractArray{R}) where {R<:Real, I<:Integer}
    ρ_avg = 0.5*(ρ[c_self] + value(ρ[c_other]))
    return -T*(p[c_self] - value(p[c_other]) + gΔz*ρ_avg)
end

@inline function half_face_two_point_kgradp(c_self::I, c_other::I, T, p::AbstractArray{R}) where {R<:Real, I<:Integer}
    return -T*(p[c_self] - value(p[c_other]))
end
