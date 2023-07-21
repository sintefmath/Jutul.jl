struct FaceMap{M, N}
    "IndirectionMap that maps cells to faces"
    cells_to_faces::M
    "IndirectionMap that maps faces to nodes"
    faces_to_nodes::M
    "Neighbors for each face"
    neighbors::Vector{N}
end

struct UnstructuredMesh{D, IM, IF, M, F} <: Jutul.JutulMesh
    faces::FaceMap{M, Tuple{Int, Int}}
    boundary_faces::FaceMap{M, Int}
    node_points::Vector{SVector{D, F}}
    index_map::IM
    face_index::IF
end

function convert_coord_points(points::AbstractMatrix{F}) where F
    dim, nn = size(points)
    @assert dim <= 3
    T_xyz = SVector{dim, eltype(points)}
    new_points = Vector{T_xyz}(undef, nn)
    for i in 1:nn
        new_points[i] = T_xyz(points[:, i])
    end
    return (new_points, dim)
end

function convert_coord_points(pts::Vector{SVector{N, F}}) where {N, F}
    return (pts, N)
end

function convert_neighborship(N::AbstractMatrix; nf = size(N, 2), nc = maximum(N), kwarg...)
    @assert size(N, 1) == 2 "Expected neighborship of (2,nf), was $(size(N))"

    N_new = Vector{Tuple{Int, Int}}(undef, nf)
    for i in 1:nf
        N_new[i] = (N[1, i], N[2, i])
    end
    return convert_neighborship(N_new; nc = nc, nf = nf, kwarg...)
end

function convert_neighborship(N::Vector{Tuple{Int, Int}}; nc = nothing, nf = nothing, allow_zero = false)
    # Just do asserts
    if !isnothing(nf)
        @assert length(N) == nf
    end
    if !isnothing(nc)
        for (f, t) in enumerate(N)
            for (c, i) in enumerate(t)
                @assert c <= nc "Neighborship exceeded $nc in face $f cell $i: neighborship was $t"
                if allow_zero
                    @assert c >= 0 "Neighborship was negative in face $f cell $i: neighborship was $t"
                else
                    @assert c > 0 "Neighborship was non-positive in face $f cell $i: neighborship was $t"
                end
            end
        end
    end
    return N
end

# Outer constructor: Take MRST format and turn into separate lists for interior and boundary
function UnstructuredMesh(cells_faces, cells_facepos, faces_nodes, faces_nodespos, node_points, face_neighbors::Matrix{Int}; kwarg...)#  where {T<:IndirectionMap, F<:Real}
    nc = length(cells_facepos)-1
    nf = length(faces_nodespos)-1
    node_points, dim = convert_coord_points(node_points)
    nn = length(node_points)

    face_neighbors = convert_neighborship(face_neighbors, nf = nf, nc = nc, allow_zero = true)
    @assert dim <= 3
    @assert dim >= 1
    # @assert maximum(face_neighbors) <= nc
    # @assert minimum(face_neighbors) >= 0
    @assert maximum(faces_nodes) <= nn "Too few nodes provided"

    new_faces_nodes = similar(faces_nodes, 0)
    new_faces_nodespos = [1]

    boundary_faces_nodes = similar(faces_nodes, 0)
    boundary_faces_nodespos = [1]

    faceindex = Int[]
    int_indices = Int[]
    bnd_indices = Int[]

    added_interior = 0
    added_boundary = 0

    for face in 1:nf
        l, r = face_neighbors[face]
        npos = faces_nodespos[face]:(faces_nodespos[face+1]-1)
        n = length(npos)
        bnd = l == 0 || r == 0
        if bnd
            for i in npos
                push!(boundary_faces_nodes, faces_nodes[i])
            end
            push!(boundary_faces_nodespos, boundary_faces_nodespos[end] + n)
            added_boundary += 1
            # Minus sign means boundary index
            push!(faceindex, -added_boundary)
            push!(bnd_indices, face)
        else
            for i in npos
                push!(new_faces_nodes, faces_nodes[i])
            end
            push!(new_faces_nodespos, new_faces_nodespos[end] + n)
            added_interior += 1
            # Positive sign for interior
            push!(faceindex, added_interior)
            push!(int_indices, face)
        end
    end
    @assert added_boundary + added_interior == nf

    new_cells_faces = similar(cells_faces, 0)
    new_cells_facepos = [1]

    boundary_cells_faces = similar(cells_faces, 0)
    boundary_cells_facepos = [1]

    for cell in 1:nc
        bnd_count = 0
        int_count = 0
        for fp in cells_facepos[cell]:(cells_facepos[cell+1]-1)
            face = cells_faces[fp]
            ix = faceindex[face]
            if ix > 0
                # Interior
                push!(new_cells_faces, ix)
                int_count += 1
            else
                # Boundary
                push!(boundary_cells_faces, abs(ix))
                bnd_count += 1
            end
        end
        push!(new_cells_facepos, new_cells_facepos[end] + int_count)
        push!(boundary_cells_facepos, boundary_cells_facepos[end] + bnd_count)
    end

    int_neighbors = face_neighbors[int_indices]
    bnd_cells = Int[]
    for i in bnd_indices
        l, r = face_neighbors[i]
        @assert l == 0 || r == 0
        push!(bnd_cells, l + r)
    end
    return UnstructuredMesh(
        new_cells_faces,
        new_cells_facepos,
        boundary_cells_faces,
        boundary_cells_facepos,
        new_faces_nodes,
        new_faces_nodespos,
        boundary_faces_nodes,
        boundary_faces_nodespos,
        node_points,
        int_neighbors,
        bnd_cells;
        kwarg...,
        face_index = faceindex
        )
end

# Middle constructor, do some checking and convert to real data structures (SVector, tuple neighbors, indirection maps)
function UnstructuredMesh(
        new_cells_faces,
        new_cells_facepos,
        boundary_cells_faces,
        boundary_cells_facepos,
        faces_nodes,
        faces_nodespos,
        boundary_faces_nodes,
        boundary_faces_nodespos,
        node_points,
        int_neighbors,
        bnd_cells;
        kwarg...
    )
    cells_to_faces = IndirectionMap(new_cells_faces, new_cells_facepos)
    cells_to_bnd = IndirectionMap(boundary_cells_faces, boundary_cells_facepos)
    faces_to_nodes = IndirectionMap(faces_nodes, faces_nodespos)
    bnd_to_nodes = IndirectionMap(boundary_faces_nodes, boundary_faces_nodespos)

    nc = length(cells_to_faces)
    nb = length(bnd_to_nodes)
    nf = length(faces_to_nodes)

    node_points, dim = convert_coord_points(node_points)
    nn = length(node_points)

    @assert dim <= 3
    @assert dim >= 1
    int_neighbors = convert_neighborship(int_neighbors)
    @assert length(bnd_cells) == nb
    bnd_cells::AbstractVector
    @assert maximum(bnd_cells) <= nc
    @assert minimum(bnd_cells) > 0

    @assert maximum(faces_to_nodes.vals) <= nn "Too few nodes provided"
    return UnstructuredMesh(cells_to_faces, cells_to_bnd, faces_to_nodes, bnd_to_nodes, node_points, int_neighbors, bnd_cells; kwarg...)
end

function UnstructuredMesh(
        cells_to_faces::T,
        cells_to_bnd::T,
        faces_to_nodes::T,
        bnd_to_nodes::T,
        node_points::Vector{SVector{dim, F}},
        face_neighbors::AbstractVector,
        boundary_cells::Vector{Int};
        index_map::IM = nothing,
        face_index::IF = nothing
    ) where {T<:IndirectionMap, IM, IF, dim, F<:Real}
    faces = FaceMap(cells_to_faces, faces_to_nodes, face_neighbors)
    bnd = FaceMap(cells_to_bnd, bnd_to_nodes, boundary_cells)
    # F = eltype(eltype(node_points))
    return UnstructuredMesh{dim, IM, IF, T, F}(faces, bnd, node_points, index_map, face_index)
end

function UnstructuredMesh(g::MRSTWrapMesh)
    G_raw = g.data
    faces_raw = Int.(vec(G_raw.cells.faces[:, 1]))
    facePos_raw = Int.(vec(G_raw.cells.facePos[:, 1]))
    nodes_raw = Int.(vec(G_raw.faces.nodes[:, 1]))
    nodePos_raw = Int.(vec(G_raw.faces.nodePos[:, 1]))
    coord = collect(G_raw.nodes.coords')
    N_raw = Int.(G_raw.faces.neighbors')
    return UnstructuredMesh(faces_raw, facePos_raw, nodes_raw, nodePos_raw, coord, N_raw)
end

