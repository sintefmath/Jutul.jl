# export half_face_flux, half_face_flux!, tp_flux, half_face_flux_kernel
export SPU, TPFA, TwoPointPotentialFlow, DarcyMassMobilityFlow, 
        CellNeighborPotentialDifference, TrivialFlow

abstract type TwoPointDiscretization <: TervDiscretization end

abstract type PotentialFlowDiscretization <: TervDiscretization end
abstract type KGradDiscretization <: PotentialFlowDiscretization end

abstract type UpwindDiscretization <: TervDiscretization end

abstract type FlowType <: TervDiscretization end
include_face_sign(::FlowType) = false

function select_primary_variables_flow_type(S, domain, system, formulation, flow_type)

end

function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type)
    
end

struct DarcyMassMobilityFlow <: FlowType end
struct DarcyMassMobilityFlowFused <: FlowType end
struct TrivialFlow <: FlowType end


function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type::Union{DarcyMassMobilityFlow, DarcyMassMobilityFlowFused})
    S[:CellNeighborPotentialDifference] = CellNeighborPotentialDifference()
    if !isa(system, SinglePhaseSystem)
        S[:RelativePermeabilities] = BrooksCoreyRelPerm(system)
    end
    S[:MassMobilities] = MassMobilities()
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

function get_connection(face, cell, faces, N, T, z, g, inc_face_sign)
    D = Dict()
    if N[1, face] == cell
        s = 1
        other = N[2, face]
    else
        s = -1
        other = N[1, face]
    end
    D[:self] = cell
    D[:other] = other
    D[:face] = face
    if inc_face_sign
        D[:face_sign] = s
    end
    if !isnothing(T)
        D[:T] = T[face]
    end
    if !isnothing(z)
        D[:gdz] = -g*(z[cell] - z[other])
    end
    return convert_to_immutable_storage(D)
end

struct TwoPointPotentialFlow{U <: Union{UpwindDiscretization, Nothing}, K <:Union{PotentialFlowDiscretization, Nothing}, F <: FlowType} <: FlowDiscretization
    upwind::U
    grad::K
    flow_type::F
    gravity::Bool
    conn_pos
    conn_data
end

function get_neighborship(grid)
    return grid.neighborship
end

function TwoPointPotentialFlow(u, k, flow_type, grid, T = nothing, z = nothing, gravity = gravity_constant)
    N = get_neighborship(grid)
    if size(N, 2) > 0
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
        get_el = (face, cell) -> get_connection(face, cell, faces, N, T, z, gravity, include_face_sign(flow_type))
        el = get_el(1, 1) # Could be junk, we just need eltype
        
        conn_data = Vector{typeof(el)}(undef, nhf)
        @threads for cell = 1:nc
            @inbounds for fpos = face_pos[cell]:(face_pos[cell+1]-1)
                conn_data[fpos] = get_el(faces[fpos], cell)
            end
        end
        @assert !isa(flow_type, TrivialFlow) "TrivialFlow only valid for grids without connections."
    else
        nc = number_of_cells(grid)
        has_grav = false
        conn_data = []
        face_pos = ones(Int64, nc+1)
    end
    TwoPointPotentialFlow{typeof(u), typeof(k), typeof(flow_type)}(u, k, flow_type, has_grav, face_pos, conn_data)
end


function select_secondary_variables_discretization!(S, domain, system, formulation, fd::TwoPointPotentialFlow)
    select_secondary_variables_flow_type!(S, domain, system, formulation, fd.flow_type)
end

function transfer(context::SingleCUDAContext, fd::TwoPointPotentialFlow{U, K, F}) where {U, K, F}
    tf = (x) -> transfer(context, x)
    u = tf(fd.upwind)
    k = tf(fd.grad)

    conn_pos = tf(fd.conn_pos)
    cd = map(tf, fd.conn_data)

    conn_data = tf(cd)

    flow_type = tf(fd.flow_type)
    has_grav = tf(fd.gravity)

    return TwoPointPotentialFlow{U, K, F}(u, k, flow_type, has_grav, conn_pos, conn_data)
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
    if mf.gravity
        @tullio pot[ph, i] = half_face_two_point_kgradp_gravity(conn_data[i], Pressure, view(PhaseMassDensities, ph, :))
    else
        @tullio pot[i] = half_face_two_point_kgradp(conn_data[i], Pressure)
    end
end

# Half face flux - trivial version which should only be used when there are no faces
function update_half_face_flux!(law, storage, model, dt, flowd::TwoPointPotentialFlow{U, K, T}) where {U,K,T<:TrivialFlow}

end

"""
Half face Darcy flux with separate potential.
"""
function update_half_face_flux!(law, storage, model, dt, flowd::TwoPointPotentialFlow{U, K, T}) where {U,K,T<:DarcyMassMobilityFlow}
    pot = storage.state.CellNeighborPotentialDifference
    mob = storage.state.MassMobilities

    flux = get_entries(law.half_face_flux_cells)
    conn_data = law.flow_discretization.conn_data
    if size(pot, 1) == 1
        # Scalar potential
        @tullio flux[ph, i] = spu_upwind_mult(conn_data[i].self, conn_data[i].other, pot[i], view(mob, ph, :))
    else
        # Multiphase potential
        @tullio flux[ph, i] = spu_upwind_mult(conn_data[i].self, conn_data[i].other, pot[ph, i], view(mob, ph, :))
    end
end

"""
Half face Darcy flux in a single fused operation (faster for immiscible models)
"""
function update_half_face_flux!(law, storage, model, dt, flowd::TwoPointPotentialFlow{U, K, T}) where {U,K,T<:DarcyMassMobilityFlowFused}
    mf = model.domain.discretizations.mass_flow

    state = storage.state
    p = state.Pressure
    mob = state.MassMobilities
    rho = state.PhaseMassDensities

    flux = get_entries(law.half_face_flux_cells)
    mf = law.flow_discretization
    conn_data = mf.conn_data
    if mf.gravity
        @tullio flux[ph, i] = half_face_two_point_flux_fused_gravity(conn_data[i], p, view(mob, ph, :), view(rho, ph, :))
    else
        @tullio flux[ph, i] = half_face_two_point_flux_fused(conn_data[i], p, view(mob, ph, :))
    end
end

"""
Perform single-point upwinding based on signed potential.
"""
@inline function spu_upwind(c_self::I, c_other::I, θ::R, λ::AbstractArray{R}) where {R<:Real, I<:Integer}
    if θ < 0
        # Flux is leaving the cell
        @inbounds λᶠ = λ[c_self]
    else
        # Flux is entering the cell
        @inbounds λᶠ = value(λ[c_other])
    end
    return λᶠ
end

"""
Perform single-point upwinding based on signed potential, then multiply the result with that potential
"""
@inline function spu_upwind_mult(c_self, c_other, θ, λ)
    λᶠ = spu_upwind(c_self, c_other, θ, λ)
    return θ*λᶠ
end

"""
Two point Darcy flux with gravity - outer version that takes in NamedTuple for static parameters
"""
@inline function half_face_two_point_kgradp_gravity(conn_data::NamedTuple, p, density)
    return half_face_two_point_kgradp_gravity(conn_data.self, conn_data.other, conn_data.T, p, conn_data.gdz, density)
end

"""
Two point Darcy flux with gravity - inner version that takes in cells and transmissibily explicitly
"""
@inline function half_face_two_point_kgradp_gravity(c_self::I, c_other::I, T, p::AbstractArray{R}, gΔz, ρ::AbstractArray{R}) where {R<:Real, I<:Integer}
    v = -T*two_point_potential_drop_half_face(c_self, c_other, p, gΔz, ρ)
    return v
end

"""
Two-point potential drop (with derivatives only respect to "c_self")
"""
@inline function two_point_potential_drop_half_face(c_self, c_other, p::AbstractVector, gΔz, ρ)
    return two_point_potential_drop(p[c_self], value(p[c_other]), gΔz, ρ[c_self], value(ρ[c_other]))
end

"""
Two-point potential drop with gravity (generic)
"""
@inline function two_point_potential_drop(p_self::Real, p_other::Real, gΔz::Real, ρ_self::Real, ρ_other::Real)
    ρ_avg = 0.5*(ρ_self + ρ_other)
    return p_self - p_other + gΔz*ρ_avg
end

"""
TPFA KGrad(p) without gravity. (Outer version, with conn_data input)
"""
@inline function half_face_two_point_kgradp(conn_data::NamedTuple, p::AbstractArray)
    half_face_two_point_kgradp(conn_data.self, conn_data.other, conn_data.T, p)
end

"""
TPFA KGrad(p) without gravity. (Inner version, with explicit inputs)
"""
@inline function half_face_two_point_kgradp(c_self::I, c_other::I, T, p::AbstractArray{R}) where {R<:Real, I<:Integer}
    return -T*(p[c_self] - value(p[c_other]))
end

"""
TPFA-SPU Mobility * KGrad(p) without gravity. (Outer version, with conn_data input)
"""
@inline function half_face_two_point_flux_fused(conn_data, p, λ)
    return half_face_two_point_flux_fused(conn_data.self, conn_data.other, conn_data.T, p, λ)
end

"""
TPFA-SPU Mobility * KGrad(p) without gravity. (Inner version, with explicit inputs)
"""
@inline function half_face_two_point_flux_fused(c_self, c_other, T, p, λ)
    θ = half_face_two_point_kgradp(c_self, c_other, T, p)
    λᶠ = spu_upwind(c_self, c_other, θ, λ)
    return λᶠ*θ
end

"""
TPFA-SPU Mobility * (KGrad(p) + G). (Outer version, with conn_data input)
"""
@inline function half_face_two_point_flux_fused_gravity(conn_data, p, λ, density)
    return half_face_two_point_flux_fused_gravity(conn_data.self, conn_data.other, conn_data.T, p, λ, conn_data.gdz, density)
end

"""
TPFA-SPU Mobility * (KGrad(p) + G). (Inner version, with explicit inputs)
"""
@inline function half_face_two_point_flux_fused_gravity(c_self, c_other, T, p, λ, gΔz, density)
    θ = half_face_two_point_kgradp_gravity(c_self, c_other, T, p, gΔz, density)
    return spu_upwind_mult(c_self, c_other, θ, λ)
end
