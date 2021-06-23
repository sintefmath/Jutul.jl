# export half_face_flux, half_face_flux!, tp_flux, half_face_flux_kernel
export SPU, TPFA, TwoPointPotentialFlow, DarcyMassMobilityFlow, 
        CellNeighborPotentialDifference, TotalMassVelocityMassFractionsFlow, FlowType

abstract type TwoPointDiscretization <: TervDiscretization end

abstract type PotentialFlowDiscretization <: TervDiscretization end
abstract type KGradDiscretization <: PotentialFlowDiscretization end

abstract type UpwindDiscretization <: TervDiscretization end

abstract type FlowType <: TervDiscretization end
include_face_sign(::FlowType) = false

function select_primary_variables_flow_type(S, domain, system, formulation, flow_type)

end

struct DarcyMassMobilityFlow <: FlowType end
struct DarcyMassMobilityFlowFused <: FlowType end


function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type::Union{DarcyMassMobilityFlow, DarcyMassMobilityFlowFused})
    S[:CellNeighborPotentialDifference] = CellNeighborPotentialDifference()
    if !isa(system, SinglePhaseSystem)
        S[:RelativePermeabilities] = BrooksCoreyRelPerm(system)
    end
    S[:MassMobilities] = MassMobilities()
end

struct TotalMassVelocityMassFractionsFlow <: FlowType end
function select_secondary_variables_flow_type!(S, domain, system, formulation, flow_type::TotalMassVelocityMassFractionsFlow)
    S[:TotalMass] = TotalMass()
end
include_face_sign(::TotalMassVelocityMassFractionsFlow) = true

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
    if nhf == 0
        conn_data = []
    else
        get_el = (face, cell) -> get_connection(face, cell, faces, N, T, z, gravity, include_face_sign(flow_type))
        el = get_el(1, 1) # Could be junk, we just need eltype
        
        conn_data = Vector{typeof(el)}(undef, nhf)
        Threads.@threads for cell = 1:nc
            @inbounds for fpos = face_pos[cell]:(face_pos[cell+1]-1)
                conn_data[fpos] = get_el(faces[fpos], cell)
            end
        end
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
    context = model.context
    if mf.gravity
        update_cell_neighbor_potential_difference_gravity!(pot, conn_data, Pressure, PhaseMassDensities, context, kernel_compatibility(context))
    else
        update_cell_neighbor_potential_difference!(pot, conn_data, Pressure, context, kernel_compatibility(context))
    end
end

# Half face flux - default reservoir version
function update_half_face_flux!(law, storage, model, dt, flowd::TwoPointPotentialFlow{U, K, T}) where {U,K,T<:DarcyMassMobilityFlow}
    pot = storage.state.CellNeighborPotentialDifference
    mob = storage.state.MassMobilities

    flux = get_entries(law.half_face_flux_cells)
    conn_data = law.flow_discretization.conn_data
    update_fluxes_from_potential_and_mobility!(flux, conn_data, pot, mob)
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

function update_cell_neighbor_potential_difference_gravity!(dpot, conn_data, p, rho, context, ::KernelAllowed)
    @kernel function kern(dpot, @Const(conn_data), @Const(p), @Const(rho))
        ph, i = @index(Global, NTuple)
        c = conn_data[i]
        rho_i = view(rho, ph, :)
        dpot[ph, i] = half_face_two_point_kgradp_gravity(c.self, c.other, c.T, p, c.gdz, rho_i)
    end
    begin
        d = size(dpot)

        kernel = kern(context.device, context.block_size, d)
        event_jac = kernel(dpot, conn_data, p, rho, ndrange = d)
        wait(event_jac)
    end
end

function update_cell_neighbor_potential_difference!(dpot, conn_data, p, context, ::KernelDisallowed)
    Threads.@threads for i in eachindex(conn_data)
        @inbounds c = conn_data[i]
        @inbounds dpot[i] = half_face_two_point_kgradp(c.self, c.other, c.T, p)
    end
end

function update_fluxes_from_potential_and_mobility!(flux, conn_data, pot, mob)
    Threads.@threads for i in eachindex(conn_data)
        @inbounds c = conn_data[i]
        for phno = 1:size(mob, 1)
            mob_i = view(mob, phno, :)
            @inbounds flux[phno, i] = spu_upwind(c.self, c.other, pot[phno, i], mob_i)
        end
    end
end

# Fused version for DarcyMassMobilityFlowFused
function update_half_face_flux!(law, storage, model, dt, flowd::TwoPointPotentialFlow{U, K, T}) where {U,K,T<:DarcyMassMobilityFlowFused}
    state = storage.state
    p = state.Pressure
    mob = state.MassMobilities
    rho = state.PhaseMassDensities

    flux = get_entries(law.half_face_flux_cells)
    conn_data = law.flow_discretization.conn_data
    update_fluxes_fused_mobility!(flux, conn_data, p, mob, rho)
end

function update_fluxes_fused_mobility!(flux, conn_data, p, mob, rho)
    Threads.@threads for i in eachindex(conn_data)
        @inbounds c = conn_data[i]
        for phno = 1:size(mob, 1)
            mob_i = view(mob, phno, :)
            rho_i = view(rho, phno, :)
            @inbounds flux[phno, i] = half_face_two_point_flux_fused(c.self, c.other, c.T, p, mob_i, c.gdz, rho_i)
        end
    end
end
# Total velocity version for TotalMassVelocityMassFractionsFlow
function update_half_face_flux!(law, storage, model, dt, flowd::TwoPointPotentialFlow{U, K, T}) where {U,K,T<:TotalMassVelocityMassFractionsFlow}
    state = storage.state
    masses = state.TotalMasses
    total = state.TotalMass
    v = state.TotalMassFlux

    flux_cells = get_entries(law.half_face_flux_cells)
    flux_faces = get_entries(law.half_face_flux_faces)

    conn_data = law.flow_discretization.conn_data
    N = model.domain.grid.neighborship
    update_fluxes_total_mass_velocity_cells!(flux_cells, conn_data, masses, total, v)
    update_fluxes_total_mass_velocity_faces!(flux_faces, N, masses, total, v)
end

function update_fluxes_total_mass_velocity_cells!(flux, conn_data, masses, total, v)
    for i in eachindex(conn_data)
        @inbounds c = conn_data[i]
        f = c.face
        for phno = 1:size(masses, 1)
            masses_i = view(masses, phno, :)
            vi = c.face_sign*v[f]
            @inbounds flux[phno, i] = half_face_fluxes_total_mass_velocity!(c.self, c.other, masses_i, total, vi)
        end
    end
end

function update_fluxes_total_mass_velocity_faces!(flux, N, masses, total, v)
    for f in 1:size(flux, 2)
        for phno = 1:size(masses, 1)
            masses_i = view(masses, phno, :)
            left = N[1, f]
            right = N[2, f]
            flux[phno, f] = half_face_fluxes_total_mass_velocity_face!(left, right, masses_i, total, v[f])
        end
    end
end

function half_face_fluxes_total_mass_velocity!(self, other, masses, total, v)
    if v < 0
        # Flux is leaving the cell
        x = masses[self]/total[self]
    else
        # Flux is entering the cell
        x = value(masses[other])/value(total[other])
    end
    return x*value(v)
end

function half_face_fluxes_total_mass_velocity_face!(left, right, masses, total, v)
    # Note the different signs. The above function (for cells) compute the half face flux
    # and recieve the signed flux going into or out of the cell. For the half face velocity
    # we have a single velocity, and the convention is to take the left cell to be upstream 
    # for a positive flux.
    if v > 0
        # Flow from left to right
        x = value(masses[left])/value(total[left])
    else
        # Flow from right to left
        x = value(masses[right])/value(total[right])
    end
    return x*v
end

# Flux primitive functions follow
@inline function spu_upwind(c_self::I, c_other::I, θ::R, λ::AbstractArray{R}) where {R<:Real, I<:Integer}
    if θ < 0
        # Flux is leaving the cell
        @inbounds λᶠ = λ[c_self]
    else
        # Flux is entering the cell
        @inbounds λᶠ = value(λ[c_other])
    end
    return λᶠ*θ
end

@inline function half_face_two_point_kgradp_gravity(c_self::I, c_other::I, T, p::AbstractArray{R}, gΔz, ρ::AbstractArray{R}) where {R<:Real, I<:Integer}
    return -T*two_point_potential_drop_half_face(c_self, c_other, p, gΔz, ρ)
end

@inline function two_point_potential_drop_half_face(c_self, c_other, p::AbstractVector, gΔz, ρ)
    return two_point_potential_drop(p[c_self], value(p[c_other]), gΔz, ρ[c_self], value(ρ[c_other]))
end

@inline function two_point_potential_drop(p_self::Real, p_other::Real, gΔz::Real, ρ_self::Real, ρ_other::Real)
    ρ_avg = 0.5*(ρ_self + ρ_other)
    return p_self - p_other + gΔz*ρ_avg
end

@inline function half_face_two_point_kgradp(c_self::I, c_other::I, T, p::AbstractArray{R}) where {R<:Real, I<:Integer}
    return -T*(p[c_self] - value(p[c_other]))
end

@inline function half_face_two_point_flux_fused(c_self::I, c_other::I, T, p::AbstractArray{R}, λ::AbstractArray{R}, gΔz, ρ::AbstractArray{R}) where {R<:Real, I<:Integer}
    θ = half_face_two_point_kgradp_gravity(c_self, c_other, T, p, gΔz, ρ)
    λᶠ = spu_upwind(c_self, c_other, θ, λ)
    return λᶠ*θ
end
