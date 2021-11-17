export DarcyMassMobilityFlow, CellNeighborPotentialDifference
struct DarcyMassMobilityFlow <: FlowType end
struct DarcyMassMobilityFlowFused <: FlowType end


struct CellNeighborPotentialDifference <: GroupedVariables end

function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type::Union{DarcyMassMobilityFlow, DarcyMassMobilityFlowFused})
    if isa(system, SinglePhaseSystem)
        S[:RelativePermeabilities] = ConstantVariables([1.0])
    else
        S[:RelativePermeabilities] = BrooksCoreyRelPerm(system)
    end
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
    # is taken with respect to cells, but there is a possibility of some cells being inactive.
    return number_of_half_faces(model.domain.discretizations.mass_flow)
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
    s = storage.state
    rho, kr, mu, p = s.PhaseMassDensities, s.RelativePermeabilities, s.PhaseViscosities, s.Pressure
    flux = get_entries(law.half_face_flux_cells)
    flow_disc = law.flow_discretization
    conn_data = flow_disc.conn_data
    pc, ref_index = capillary_pressure(model, s)
    if flow_disc.gravity
        # Multiphase potential
        @tullio flux[ph, i] = get_immiscible_flux_gravity(conn_data[i], ph, p, rho, kr, mu, pc, ref_index)
    else
        # Scalar potential
        @tullio flux[ph, i] = get_immiscible_flux_no_gravity(conn_data[i], ph, p, rho, kr, mu)
    end
end

function get_immiscible_flux_gravity(cd, ph, p, rho, kr, mu, pc, ref_index)
    c, i, gΔz, T = cd.self, cd.other, cd.gdz, cd.T
    ∂ = (x) -> local_ad(x, c)
    return immiscible_flux_gravity(c, i, ph, ∂(kr), ∂(mu), ∂(rho), ∂(p), ∂(pc), T, gΔz, ref_index)
end

function immiscible_flux_gravity(c, i, ph, kᵣ, μ, ρ, P, pc, T, gΔz, ref_index)
    @inbounds ρ_c = ρ[ph, c]
    @inbounds ρ_i = ρ[ph, i]
    ρ_avg = 0.5*(ρ_i + ρ_c)
    Δpc = capillary_gradient(pc, c, i, ph, ref_index)
    @inbounds Δp = P[c] - P[i] + Δpc
    θ = -T*(Δp + gΔz*ρ_avg)
    if θ < 0
        # Flux is leaving the cell
        @inbounds ρλᶠ = ρ_c*kᵣ[ph, c]/μ[ph, c]
    else
        # Flux is entering the cell
        @inbounds ρλᶠ = ρ_i*kᵣ[ph, i]/μ[ph, i]
    end
    return ρλᶠ*θ
end

function get_immiscible_flux_no_gravity(cd, ph, p, rho, kr, mu)
    c, i, T = cd.self, cd.other, cd.T
    ∂ = (x) -> local_ad(x, c)
    return immiscible_flux_no_gravity(c, i, ph, ∂(kr), ∂(mu), ∂(rho), ∂(p), T)
end

function immiscible_flux_no_gravity(c, i, ph, kᵣ, μ, ρ, P, T)
    @inbounds θ = -T*(P[c] - P[i])
    if θ < 0
        # Flux is leaving the cell
        up_c = c
    else
        # Flux is entering the cell
        up_c = i
    end
    @inbounds ρλᶠ = ρ[ph, up_c]*kᵣ[ph, up_c]/μ[ph, up_c]
    return ρλᶠ*θ
end


capillary_gradient(::Nothing, c_l, c_r, ph, ph_ref) = 0.0
function capillary_gradient(pc, c_l, c_r, ph, ph_ref)
    if ph == ph_ref
        Δp_c = 0.0
    elseif ph < ph_ref
        Δp_c = pc[ph, c_l] - pc[ph, c_r]
    else
        Δp_c = pc[ph-1, c_l] - pc[ph-1, c_r]
    end
end


"""
Half face Darcy flux with separate potential. (Compositional version)
"""
function update_half_face_flux!(law, storage, model::SimulationModel{D, S}, dt, flowd::TwoPointPotentialFlow{U, K, T}) where {D,S<:TwoPhaseCompositionalSystem,U,K,T<:DarcyMassMobilityFlow}
    state = storage.state
    pot = state.CellNeighborPotentialDifference
    X = state.LiquidMassFractions
    Y = state.VaporMassFractions
    kr = state.RelativePermeabilities
    μ = state.PhaseViscosities
    ρ = state.PhaseMassDensities
    fr = state.FlashResults

    flux = get_entries(law.half_face_flux_cells)
    conn_data = law.flow_discretization.conn_data

    if size(pot, 1) == 1
        # Scalar potential
        for i in 1:length(pot)
            @views compositional_flux_single_pot!(flux[:, i], fr, conn_data[i], pot[1, i], X, Y, ρ, kr, μ)
        end
    else
        error()
        # Multiphase potential
        @tullio flux[c, i] = spu_upwind_mult_index(conn_data[i], ph, pot[ph, i], mob)
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


function compositional_flux_single_pot!(q, flash_state, conn_data, θ, X, Y, ρ, kr, μ)
    if θ < 0
        ix = conn_data.self
        f = (x) -> x
    else
        ix = conn_data.other
        f = value
    end
    s = flash_state[ix].state
    compositional_flux_single_pot!(q, s, conn_data, ix, f, θ, X, Y, ρ, kr, μ)
end

function compositional_flux_single_pot!(q, ::TwoPhaseLiquidVapor, conn_data, ix, f, θ, X, Y, ρ, kr, μ)
    kr_l = f(kr[1, ix])
    kr_v = f(kr[2, ix])
    ρ_l = f(ρ[1, ix])
    ρ_v = f(ρ[2, ix])
    μ_l = f(μ[1, ix])
    μ_v = f(μ[2, ix])

    for i in eachindex(q)
        q[i] = ((kr_l/μ_l)*f(X[i, ix])*ρ_l + (kr_v/μ_v)*f(Y[i, ix])*ρ_v)*θ
    end
end

function compositional_flux_single_pot!(q, ::SinglePhaseLiquid, conn_data, ix, f, θ, X, Y, ρ, kr, μ)
    kr_l = f(kr[1, ix])
    ρ_l = f(ρ[1, ix])
    μ_l = f(μ[1, ix])

    for i in eachindex(q)
        q[i] = (kr_l/μ_l)*f(X[i, ix])*ρ_l*θ
    end
end

function compositional_flux_single_pot!(q, ::SinglePhaseVapor, conn_data, ix, f, θ, X, Y, ρ, kr, μ)
    kr_v = f(kr[2, ix])
    ρ_v = f(ρ[2, ix])
    μ_v = f(μ[2, ix])

    for i in eachindex(q)
        q[i] = (kr_v/μ_v)*f(Y[i, ix])*ρ_v*θ
    end
end
