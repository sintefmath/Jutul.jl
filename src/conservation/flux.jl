export SPU, TPFA, TwoPointPotentialFlow, TrivialFlow, get_neighborship

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

struct TrivialFlow <: FlowType end

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

function remap_connection(conn, self, other, face)
    D = Dict()
    for k in keys(conn)
        if k == :self
            newval = self
        elseif k == :other
            newval = other
        elseif k == :face
            newval = face
        else
            newval = conn[k]
        end
        D[k] = newval
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
        @batch for cell = 1:nc
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

function subdiscretization(disc::TwoPointPotentialFlow, subg, mapper::FiniteVolumeGlobalMap)
    u, k, flow_type, has_grav = disc.upwind, disc.grad, disc.flow_type, disc.gravity

    face_pos_global, conn_data_global = disc.conn_pos, disc.conn_data
    N = get_neighborship(subg)
    faces, face_pos = get_facepos(N)

    T = eltype(conn_data_global)
    conn_data = Vector{T}(undef, length(faces))
    for c in 1:number_of_cells(subg)
        c_g = global_cell(c, mapper)
        # Loop over half-faces for this cell
        for f_p in face_pos[c]:face_pos[c+1]-1
            f = faces[f_p]
            f_g = global_face(f, mapper)
            # Loop over the corresponding global half-faces
            for f_i in face_pos_global[c_g]:face_pos_global[c_g+1]-1
                conn = conn_data_global[f_i]
                if conn.face == f_g
                    # verify that this is actually the right global cell!
                    @assert conn.self == c_g
                    other = local_cell(conn.other, mapper)
                    conn_data[f_p] = remap_connection(conn, c, other, f)
                    break
                end
            end
        end
    end
    return TwoPointPotentialFlow{typeof(u), typeof(k), typeof(flow_type)}(u, k, flow_type, has_grav, face_pos, conn_data)
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


# Half face flux - trivial version which should only be used when there are no faces
function update_half_face_flux!(law, storage, model, dt, flowd::TwoPointPotentialFlow{U, K, T}) where {U,K,T<:TrivialFlow}

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

@inline function spu_upwind_index(c_self::I, c_other::I, index::I, θ::R, λ::AbstractArray{R}) where {R<:Real, I<:Integer}
    if θ < 0
        # Flux is leaving the cell
        @inbounds λᶠ = λ[index, c_self]
    else
        # Flux is entering the cell
        @inbounds λᶠ = value(λ[index, c_other])
    end
    return λᶠ
end

"""
Perform single-point upwinding based on signed potential, then multiply the result with that potential
"""
@inline function spu_upwind_mult_index(c, index, θ, λ)
    λᶠ = spu_upwind_index(c.self, c.other, index, θ, λ)
    return θ*λᶠ
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
