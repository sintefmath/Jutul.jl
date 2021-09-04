export DarcyMassMobilityFlow, CellNeighborPotentialDifference
struct DarcyMassMobilityFlow <: FlowType end
struct DarcyMassMobilityFlowFused <: FlowType end


struct CellNeighborPotentialDifference <: GroupedVariables end

function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type::Union{DarcyMassMobilityFlow, DarcyMassMobilityFlowFused})
    S[:CellNeighborPotentialDifference] = CellNeighborPotentialDifference()
    if !isa(system, SinglePhaseSystem)
        S[:RelativePermeabilities] = BrooksCoreyRelPerm(system)
    end
    S[:MassMobilities] = MassMobilities()
end


function single_unique_potential(model)
    # We should add capillary pressure here ventually
    return model.domain.discretizations.mass_flow.gravity
end

function degrees_of_freedom_per_entity(model, sf::CellNeighborPotentialDifference)
    if single_unique_potential(model)
        n = number_of_phases(model.system)
    else
        n = 1
    end
    return n
end

function associated_entity(::CellNeighborPotentialDifference)
    Cells()
end

function number_of_entities(model, pv::CellNeighborPotentialDifference)
    # We have two entities of potential difference per face of the domain since the difference
    # is taken with respect to cells
    return 2*count_entities(model.domain, Faces())
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
        @tullio flux[ph, i] = spu_upwind_mult_index(conn_data[i], ph, pot[1, i], mob)
    else
        # Multiphase potential
        @tullio flux[ph, i] = spu_upwind_mult_index(conn_data[i], ph, pot[ph, i], mob)
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
