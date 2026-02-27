export SPU, TPFA, TwoPointPotentialFlowHardCoded, get_neighborship

abstract type TwoPointDiscretization <: JutulDiscretization end

export PotentialFlowDiscretization, TwoPointDiscretization, KGradDiscretization, UpwindDiscretization

abstract type PotentialFlowDiscretization <: JutulDiscretization end
abstract type KGradDiscretization <: PotentialFlowDiscretization end

abstract type UpwindDiscretization <: JutulDiscretization end

"""
Two-point flux approximation.
"""
struct TPFA{T} <: KGradDiscretization
    left::T
    right::T
    face_sign::T
end

function subdiscretization(tpfa::TPFA, subg, mapper::Jutul.FiniteVolumeGlobalMap, face)
    (; left, right, face_sign) = tpfa
    gmap = mapper.global_to_local
    return TPFA(gmap[left], gmap[right], face_sign)
end

function cell_pair(tpfa::TPFA)
    return (tpfa.left, tpfa.right)
end

"""
Single-point upwinding.
"""
struct SPU{T} <: UpwindDiscretization
    left::T
    right::T
end

function subdiscretization(spu::SPU, subg, mapper::Jutul.FiniteVolumeGlobalMap, face)
    (; left, right) = spu
    gmap = mapper.global_to_local
    return SPU(gmap[left], gmap[right])
end

function cell_pair(spu::SPU)
    return (spu.left, spu.right)
end

function discretization_stencil(d::Union{TPFA, SPU}, ::Cells)
    return cell_pair(d)
end

export PotentialFlow
struct PotentialFlow{AD, K, U, HF} <: FlowDiscretization
    kgrad::K
    upwind::U
    half_face_map::HF
    function PotentialFlow(kgrad::K, upwind::U, hf::HF; ad::Symbol = :generic) where {K, U, HF}
        @assert ad in (:fvm, :generic)
        return new{ad, K, U, HF}(kgrad, upwind, hf)
    end
end

function PotentialFlow(g::JutulMesh; kwarg...)
    N = get_neighborship(g)
    nc = number_of_cells(g)
    return PotentialFlow(N, nc; kwarg...)
end

function PotentialFlow(N::AbstractMatrix, nc = maximum(N); kgrad = nothing, upwind = nothing, ad = :generic)
    nf = size(N, 2)
    hf = half_face_map(N, nc)
    T = eltype(N)
    if isnothing(kgrad) || kgrad == :tpfa
        kgrad = Vector{TPFA{T}}(undef, nf)
        for i in 1:nf
            left = N[1, i]
            right = N[2, i]
            face_sign = 1
            kgrad[i] = TPFA(left, right, face_sign)
        end
    end
    if isnothing(upwind) || upwind == :spu
        upwind = Vector{SPU{T}}(undef, nf)
        for i in 1:nf
            left = N[1, i]
            right = N[2, i]
            upwind[i] = SPU(left, right)
        end
    end
    @assert upwind isa AbstractVector
    @assert kgrad isa AbstractVector
    return PotentialFlow(kgrad, upwind, hf, ad = ad)
end

function subdiscretization(disc::PotentialFlow{ad}, subg, mapper::FiniteVolumeGlobalMap) where ad
    # kgrad
    # upwind
    # half_face_map -> N -> remap N -> half_face_map
    N, nc = half_face_map_to_neighbors(disc.half_face_map)

    faces = mapper.faces
    N = N[:, faces]
    # Remap cells in N
    for (i, c) in enumerate(N)
        N[i] = mapper.global_to_local[c]
    end
    kgrad = disc.kgrad[faces]
    upwind = disc.upwind[faces]
    for i in eachindex(kgrad, upwind, faces)
        kgrad[i] = subdiscretization(kgrad[i], subg, mapper, faces[i])
        upwind[i] = subdiscretization(upwind[i], subg, mapper, faces[i])
    end
    hf = half_face_map(N, nc)
    return PotentialFlow(kgrad, upwind, hf, ad = ad)
end

function local_discretization(eq::ConservationLaw{S, D, FT, N}, i) where {S, D<:PotentialFlow, FT, N}
    disc = eq.flow_discretization
    face_map = local_half_face_map(disc.half_face_map, i)
    div = (x, F) -> divergence!(x, F, face_map)
    div = F -> divergence(F, face_map)
    face_disc = (face) -> (kgrad = disc.kgrad[face], upwind = disc.upwind[face])
    return (div! = div, div = div, face_disc)
end

function get_connection(face, cell, N, inc_face_sign)
    if N[1, face] == cell
        s = 1
        other = N[2, face]
    else
        s = -1
        other = N[1, face]
    end
    if inc_face_sign
        out = (self = cell, other = other, face = face, face_sign = s)
    else
        out = (self = cell, other = other, face = face)
    end
    return out
end

function remap_connection(conn::T, self::I, other::I, face::I) where {T, I<:Integer}
    vals = values(conn)
    i = 1
    for k in keys(conn)
        if k == :self
            vals = setindex(vals, self, i)
        elseif k == :other
            vals = setindex(vals, other, i)
        elseif k == :face
            vals = setindex(vals, face, i)
        end
        i += 1
    end
    conn = (; zip(keys(conn), vals)...)::T
    return conn
end

struct TwoPointPotentialFlowHardCoded{C, D} <: FlowDiscretization
    gravity::Bool
    conn_pos::C
    conn_data::D
end

function TwoPointPotentialFlowHardCoded(grid::JutulMesh)
    N = get_neighborship(grid)
    return TwoPointPotentialFlowHardCoded(N, number_of_cells(grid))
end

function TwoPointPotentialFlowHardCoded(N::AbstractMatrix, nc = maximum(N, init = 1))
    if size(N, 2) > 0
        faces, face_pos = get_facepos(N, nc)
        nhf = length(faces)
        @assert length(face_pos) - 1 == nc
        get_el = (face, cell) -> get_connection(face, cell, N, true)
        el = get_el(1, 1) # Could be junk, we just need eltype
        conn_data = Vector{typeof(el)}(undef, nhf)
        @batch for cell = 1:nc
                @inbounds for fpos = face_pos[cell]:(face_pos[cell+1]-1)
                conn_data[fpos] = get_el(faces[fpos], cell)
            end
        end
    else
        conn_data = []
        face_pos = ones(Int64, nc+1)
    end
    return TwoPointPotentialFlowHardCoded{typeof(face_pos), typeof(conn_data)}(true, face_pos, conn_data)
end

function local_discretization(eq::ConservationLaw{S, D, FT, N}, i) where {S, D<:TwoPointPotentialFlowHardCoded, FT, N}
    disc = eq.flow_discretization
    start = disc.conn_pos[i]
    stop = disc.conn_pos[i+1]-1
    return view(disc.conn_data, start:stop)
end

function update_equation_in_entity!(eq_buf::AbstractVector{T_e}, self_cell, state, state0, eq::ConservationLaw{S, D, FT, N}, model, Δt, ldisc = local_discretization(eq, self_cell)) where {T_e, S, D<:TwoPointPotentialFlowHardCoded, FT<:Jutul.FluxType, N}
    # This will be called if a local TPFA code is used with some kind of general
    # AD, e.g. in adjoints.
    # Compute accumulation term
    conserved = conserved_symbol(eq)
    M₀ = state0[conserved]
    M = state[conserved]
    for i in eachindex(eq_buf)
        @inbounds eq_buf[i] = accumulation_term(M, M₀, Δt, i, self_cell)
    end
    # Compute ∇⋅V
    if length(ldisc) > 0
        flux(ld) = face_flux(ld.self, ld.other, ld.face, ld.face_sign, eq, state, model, Δt, eq.flow_discretization, Val(T_e))
        div_v = flux(ldisc[1])
        for i in 2:length(ldisc)
            ld = ldisc[i]
            q_i = flux(ld)
            div_v += q_i
        end
        for i in eachindex(div_v)
            @inbounds eq_buf[i] += div_v[i]
        end
    end
    return eq_buf
end

number_of_half_faces(tp::TwoPointPotentialFlowHardCoded) = length(tp.conn_data)

function get_neighborship(grid; internal = true)
    @assert internal
    return grid.neighborship
end


function subdiscretization(disc::TwoPointPotentialFlowHardCoded, subg, mapper::FiniteVolumeGlobalMap)
    has_grav = disc.gravity

    face_pos_global, conn_data_global = disc.conn_pos, disc.conn_data
    N = get_neighborship(subg)
    nc = number_of_cells(subg)
    faces, face_pos = get_facepos(N, nc)

    T = eltype(conn_data_global)
    nc = length(mapper.inner_to_full_cells)
    counts = compute_counts_subdisc(face_pos, faces, face_pos_global, conn_data_global, mapper, nc)
    next_face_pos = cumsum([1; counts])

    conn_data = conn_data_subdisc(face_pos, faces, face_pos_global, next_face_pos, conn_data_global::Vector{T}, mapper, nc)
    face_pos = next_face_pos
    # face_pos = new_offsets
    # conn_data = vcat(new_conn...)
    return TwoPointPotentialFlowHardCoded{typeof(face_pos), typeof(conn_data)}(has_grav, face_pos, conn_data)
end

function compute_counts_subdisc(face_pos, faces, face_pos_global, conn_data_global, mapper, nc)
    counts = zeros(Int64, nc)
    for local_cell_no in 1:nc
        # Map inner index -> full index
        c = full_cell(local_cell_no, mapper)
        c_g = global_cell(c, mapper)
        counter = 0
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
                    counter += 1
                    break
                end
            end
        end
        counts[local_cell_no] = counter
    end
    return counts
end

function conn_data_subdisc(face_pos, faces, face_pos_global, next_face_pos, conn_data_global::Vector{T}, mapper, nc) where T
    conn_data = Vector{T}(undef, next_face_pos[end]-1)
    touched = BitVector(false for i = 1:length(conn_data))
    for local_cell_no in 1:nc
        # Map inner index -> full index
        c = full_cell(local_cell_no, mapper)
        c_g = global_cell(c, mapper)
        counter = 0
        start = face_pos[c]
        stop = face_pos[c+1]-1
        hf_offset = next_face_pos[local_cell_no]-1
        # Loop over half-faces for this cell
        for f_p in start:stop
            f = faces[f_p]
            f_g = global_face(f, mapper)
            done = false
            # Loop over the corresponding global half-faces
            for f_i in face_pos_global[c_g]:face_pos_global[c_g+1]-1
                conn = conn_data_global[f_i]::T
                # @info "$f_i" conn.face f_g f
                if conn.face == f_g
                    # verify that this is actually the right global cell!
                    @assert conn.self == c_g
                    counter += 1
                    other = local_cell(conn.other, mapper) # bad performance??
                    conn_data[hf_offset + counter] = remap_connection(conn, c, other, f)::T
                    touched[hf_offset + counter] = true
                    done = true
                    # @info conn_data[hf_offset + counter]
                    break
                end
            end
            @assert done
        end
    end
    @assert all(touched) "Only $(count(touched))/$(length(touched)) were kept? Something is wrong."
    return conn_data
end

function transfer(context::SingleCUDAContext, fd::TwoPointPotentialFlowHardCoded)
    tf = (x) -> transfer(context, x)

    conn_pos = tf(fd.conn_pos)
    cd = map(tf, fd.conn_data)

    conn_data = tf(cd)
    has_grav = tf(fd.gravity)

    return TwoPointPotentialFlowHardCoded{typeof(conn_pos), typeof(conn_data)}(has_grav, conn_pos, conn_data)
end

"""
Two-point potential drop with gravity (generic)
"""
@inline function two_point_potential_drop(p_self::Real, p_other::Real, gΔz::Real, ρ_self::Real, ρ_other::Real)
    ρ_avg = 0.5*(ρ_self + ρ_other)
    return p_self - p_other + gΔz*ρ_avg
end

export upw_flux
function upw_flux(v, l, r)
    if v > 0
        # Flow l -> r
        out = l
    else
        out = r
    end
    return out
end

function upw_flux(v, l::T, r::T) where {T<:ST.ADval}
    if v > 0
        out = l + r*0
    else
        out = r + l*0
    end
    return out
end
