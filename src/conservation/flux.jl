export SPU, TPFA, TwoPointPotentialFlowHardCoded, TrivialFlow, get_neighborship

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

"""
Single-point upwinding.
"""
struct SPU{T} <: UpwindDiscretization
    left::T
    right::T
end

export PotentialFlow
struct PotentialFlow{K, U, HF} <: FlowDiscretization
    kgrad::K
    upwind::U
    half_face_map::HF
    function PotentialFlow(kgrad::K, upwind::U, hf::HF) where {K, U, HF}
        return new{K, U, HF}(kgrad, upwind, hf)
    end
end

function PotentialFlow(g::JutulMesh; kwarg...)
    N = get_neighborship(g)
    nc = number_of_cells(g)
    PotentialFlow(N, nc; kwarg...)
end

function PotentialFlow(N::AbstractMatrix, nc = maximum(N); kgrad = nothing, upwind = nothing)
    nf = size(N, 2)
    hf = half_face_map(N, nc)
    T = eltype(N)
    if isnothing(kgrad)
        kgrad = Vector{TPFA{T}}(undef, nf)
        for i in 1:nf
            left = N[1, i]
            right = N[2, i]
            face_sign = 1
            kgrad[i] = TPFA(left, right, face_sign)
        end
    end
    if isnothing(upwind)
        upwind = Vector{SPU{T}}(undef, nf)
        for i in 1:nf
            left = N[1, i]
            right = N[2, i]
            upwind[i] = SPU(left, right)
        end
    end
    return PotentialFlow(kgrad, upwind, hf)
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

    return convert_to_immutable_storage(D)
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

struct TwoPointPotentialFlowHardCoded{C, D, F} <: FlowDiscretization
    gravity::Bool
    conn_pos::C
    conn_data::D
    face_to_half_face::F
    face_neighborship::F
end

function TwoPointPotentialFlowHardCoded(grid::JutulMesh)
    N = get_neighborship(grid)
    return TwoPointPotentialFlowHardCoded(N, number_of_cells(grid))
end

function TwoPointPotentialFlowHardCoded(N::AbstractMatrix, nc = maximum(N))
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
        fn = map(i -> (N[1, i], N[2, i]), 1:size(N, 2))

        function findface(cell, face)
            for fpos = face_pos[cell]:(face_pos[cell+1]-1)
                if conn_data[fpos].face == face
                    return fpos
                end
            end
            error()
        end

        f2hf = similar(fn)
        for i in eachindex(f2hf)
            l, r = fn[i]
            f2hf[i] = (findface(l, i), findface(r, i))
        end
    else
        nc = number_of_cells(grid)
        conn_data = []
        face_pos = ones(Int64, nc+1)
        f2hf = Vector{Tuple{Int64, Int64}}() # tuple of positions in half face map
        fn = similar(f2hf) # tuple of left, right face
    end
    return TwoPointPotentialFlowHardCoded{typeof(face_pos), typeof(conn_data), typeof(f2hf)}(true, face_pos, conn_data, f2hf, fn)
end

number_of_half_faces(tp::TwoPointPotentialFlowHardCoded) = length(tp.conn_data)

function get_neighborship(grid)
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
