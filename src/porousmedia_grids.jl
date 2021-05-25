export MinimalTPFAGrid, TPFAHalfFaceData
export get_cell_faces, get_facepos, get_cell_neighbors
export number_of_cells, number_of_faces, number_of_half_faces

export TwoPointFlux, SinglePointUpstream

export transfer

# Helpers follow
"Minimal struct for TPFA connections (transmissibility + dz + cell pair)"
struct TPFAHalfFaceData{R<:Real,I<:Integer}
    T::R
    dz::R
    self::I
    other::I
end

function TPFAHalfFaceData{R, I}(target::TPFAHalfFaceData) where {R<:Real, I<:Integer}
    return TPFAHalfFaceData(R(target.T), R(target.dz), I(target.self), I(target.other))
end

abstract type PorousMediumGrid <: TervGrid end
abstract type ReservoirGrid <: PorousMediumGrid end
# TPFA grid
"Minimal struct for TPFA-like grid. Just connection data and pore-volumes"
struct MinimalTPFAGrid{R<:AbstractFloat, I<:Integer} <: ReservoirGrid
    pore_volumes::AbstractVector{R}
    neighborship::AbstractArray{I}
    # conn_data::AbstractArray{TPFAHalfFaceData{R,I}}
    # conn_pos::AbstractArray{I}
    # pv::AbstractArray{R}
    function MinimalTPFAGrid(pv, N)
        @assert size(N, 1) == 2
        @assert all(pv .> 0) "Pore volumes must be positive"
        new{eltype(pv), eltype(N)}(pv, N)
    end
end

struct SinglePointUpstream <: TervDiscretization

end

struct TwoPointFlux <: TervDiscretization
    conn_data
    conn_pos
    function TwoPointFlux(conn_data, nc)
        cno = [i.self for i in conn_data]
        # Slow code for the same thing:
        # counts = [sum(cno .== j) for j in 1:length(pv)]
        # nc = length(pv)
        counts = similar(cno, nc)
        index = 1
        cell = 1
    
        while cell < nc + 1
            count = 0
            while cno[index] == cell
                count += 1
                index += 1
                if index > length(cno)
                    break
                end
            end
            counts[cell] = count
            cell += 1
        end
        cpos = cumsum(vcat([1], counts))
        new(conn_data, cpos)
    end    
end


function declare_units(G::MinimalTPFAGrid)
    c = (Cells(), length(G.pore_volumes))  # Cells equal to number of pore volumes
    f = (Faces(), size(G.neighborship, 2)) # Faces
    return [c, f]
end

function get_cell_faces(N, nc = nothing)
    # Create array of arrays where each entry contains the faces of that cell
    t = eltype(N)
    if length(N) == 0
        cell_faces = ones(t, 1)
    else
        if isnothing(nc)
            nc = maximum(N)
        end
        cell_faces = [Vector{t}() for i = 1:nc]
        for i in 1:size(N, 1)
            for j = 1:size(N, 2)
                push!(cell_faces[N[i, j]], j)
            end
        end
        # Sort each of them
        for i in cell_faces
            sort!(i)
        end
    end
    return cell_faces
end

function get_cell_neighbors(N, nc = maximum(N), includeSelf = true)
    # Find faces in each array
    t = typeof(N[1])
    cell_neigh = [Vector{t}() for i = 1:nc]
    for i in 1:size(N, 2)
        push!(cell_neigh[N[1, i]], N[2, i])
        push!(cell_neigh[N[2, i]], N[1, i])
    end
    # Sort each of them
    for i in eachindex(cell_neigh)
        loc = cell_neigh[i]
        if includeSelf
            push!(loc, i)
        end
        sort!(loc)
    end
    return cell_neigh
end

function get_facepos(N)
    if length(N) == 0
        t = eltype(N)
        faces = zeros(t, 0)
        facePos = ones(t, 2)
    else
        cell_faces = get_cell_faces(N)
        counts = [length(x) for x in cell_faces]
        facePos = cumsum([1; counts])
        faces = reduce(vcat, cell_faces)
    end
    return (faces, facePos)
end

function transfer(context::SingleCUDAContext, domain)
    error("Not yet reimplemented")
    F = context.float_t
    I = context.index_t
    # Next line should be something like
    # conn_data = CuArray{TPFAHalfFaceData{F, I}}(grid.conn_data)
    # once convert is implemented...
    conn_data = CuArray(TPFAHalfFaceData{F, I}.(grid.conn_data))
    pv = CuArray{F}(grid.pv)
    conn_pos = CuArray{I}(grid.conn_pos)
    return MinimalTPFADomain(conn_data, conn_pos, pv)
end

function transfer(::DefaultContext, grid)
    return grid
end
