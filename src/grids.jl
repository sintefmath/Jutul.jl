export MRSTGrid, MinimalTPFAGrid, TPFAHalfFaceData
export get_cell_faces, get_facepos, get_cell_neighbors
export number_of_cells, number_of_faces, number_of_half_faces


# Helpers follow
"Minimal struct for TPFA connections (transmissibility + dz + cell pair)"
struct TPFAHalfFaceData{R<:Real,I<:Integer}
    T::R
    dz::R
    self::I
    other::I
end

# TPFA grid
"Minimal struct for TPFA-like grid. Just connection data and pore-volumes"
struct MinimalTPFAGrid{R<:AbstractFloat, I<:Integer} <: TervGrid
    conn_data::AbstractArray{TPFAHalfFaceData{R,I}}
    pv::AbstractArray{R}
end


# Member functions, TPFA grid
function number_of_cells(G::MinimalTPFAGrid)
    return length(G.pv)
end

function number_of_faces(G::MinimalTPFAGrid)
    return div(number_of_half_faces(G), 2)
end

function number_of_half_faces(G::MinimalTPFAGrid)
    return length(G.conn_data)
end

function get_pore_volume(G::MinimalTPFAGrid)
    return G.pore_volume
end

function get_cell_faces(N, nc = maximum(N))
    # Create array of arrays where each entry contains the faces of that cell
    t = typeof(N[1])
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
    cell_faces = get_cell_faces(N)
    
    counts = [length(x) for x in cell_faces]
    facePos = cumsum([1; counts])
    faces = reduce(vcat, cell_faces)
    return (faces, facePos)
end
