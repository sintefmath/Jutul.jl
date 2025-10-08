function Jutul.triangulate_mesh(m::UnstructuredMesh{D}; outer = false, flatten = true) where D
    pts = Vector{SVector{D, Float64}}()
    tri = Vector{SVector{3, Int64}}()
    cell_index = Vector{Int64}()
    face_index = Vector{Int64}()

    offset = 0

    dest = (cell_index, face_index, pts, tri)
    for d in dest
        # Assume hexahedral, 6 faces per cell, triangulated into 4 parts each
        sizehint!(d, 24*number_of_cells(m))
    end
    if D == 2
        cell_centroids = SVector{D, Float64}[]
        nc = number_of_cells(m)
        for i in 1:nc
            c, v = compute_centroid_and_measure(m, Cells(), i)
            push!(cell_centroids, c)
        end
    else
        # Not needed.
        cell_centroids = missing
    end

    add_points!(e, e_def, offset, face_offset) = triangulate_and_add_faces!(dest, m, e, e_def, cell_centroids, offset = offset, face_offset = face_offset)
    if !outer
        offset = add_points!(Faces(), m.faces, offset, 0)
    end
    face_offset = number_of_faces(m)
    offset = add_points!(BoundaryFaces(), m.boundary_faces, offset, face_offset)

    if flatten
        pts = plot_flatten_helper(pts)
        tri = plot_flatten_helper(tri)
    end
    cell_buffer = zeros(length(cell_index))
    face_buffer = zeros(length(face_index))

    mapper = (
                Cells = (cell_data) -> mesh_data_to_tris!(cell_buffer, cell_data, cell_index)::Vector{Float64},
                Faces = (face_data) -> mesh_data_to_tris!(face_buffer, face_data, face_index)::Vector{Float64},
                indices = (Cells = cell_index, Faces = face_index)
            )
    return (points = pts, triangulation = tri, mapper = mapper)
end

function mesh_data_to_tris!(out::Vector{Float64}, cell_data, cell_index)
    n = length(cell_index)
    @assert length(out) == n
    for i in eachindex(cell_index)
        c = cell_index[i]
        @inbounds out[i] = cell_data[c]
    end
    return out::Vector{Float64}
end

function triangulate_and_add_faces!(dest, m, e, faces, cell_centroids; offset = 0, face_offset = 0)
    node_pts = m.node_points
    T = eltype(node_pts)
    for f in 1:count_entities(m, e)
        C = zero(T)
        nodes = faces.faces_to_nodes[f]
        n = length(nodes)
        for node in nodes
            C += node_pts[node]
        end
        C /= n
        offset = triangulate_and_add_faces!(dest, f + face_offset, faces.neighbors[f], C, nodes, node_pts, cell_centroids, n; offset = offset)
    end
    return offset
end

function triangulate_and_add_faces!(dest, face, neighbors, C, nodes, node_pts::Vector{SVector{2, T}}, cell_centroids, n; offset = 0) where {T}
    cell_index, face_index, pts, tri = dest
    new_vert_count = n + 1
    @assert new_vert_count == 3
    for cell in neighbors
        if cell > 0
            for i in 1:new_vert_count
                push!(cell_index, cell)
                push!(face_index, face)
                push!(pts, svector_local_point(cell_centroids[cell], i-1, nodes, node_pts))
            end
            for i in 1:n
                push!(tri, svector_cyclical_tesselation(n, i, offset))
            end
            offset = offset + new_vert_count
        end
    end
    return offset
end


function triangulate_and_add_faces!(dest, face, neighbors, C, nodes, node_pts::Vector{SVector{3, T}}, cell_centroids, n; offset = 0) where {T}
    cell_index, face_index, pts, tri = dest
    if n == 3
        # Just add the triangle
        for cell in neighbors
            for i in 1:n
                push!(cell_index, cell)
                push!(face_index, face)
                push!(pts, node_pts[nodes[i]])
            end
            push!(tri, SVector{3, Int}(offset+1, offset + 2, offset + 3))
            offset += n
        end
    elseif n == 4
        # Tesselation is known
        for cell in neighbors
            for i in 1:n
                push!(cell_index, cell)
                push!(face_index, face)
                push!(pts, node_pts[nodes[i]])
            end
            push!(tri, SVector{3, Int}(offset+1, offset + 2, offset + 3))
            push!(tri, SVector{3, Int}(offset+3, offset + 4, offset + 1))
            offset += n
        end
    else
        new_vert_count = n + 1
        for cell in neighbors
            if cell > 0
                for i in 1:new_vert_count
                    push!(cell_index, cell)
                    push!(face_index, face)
                    push!(pts, svector_local_point(C, i-1, nodes, node_pts))
                end
                for i in 1:n
                    push!(tri, svector_cyclical_tesselation(n, i, offset))
                end
                offset = offset + new_vert_count
            end
        end
    end
    return offset
end

function svector_cyclical_tesselation(n::Int, i::Int, offset::Int)
    # Create a triangulation of face, assuming convexity
    # Each tri is two successive points on boundary connected to centroid
    # First column i + 1
    # Second column: Always the center point
    # Third column: Wrap-around to first column
    if i == 1
        t = n + offset + 1
    else
        t = i + offset
    end
    return @SVector [i + 1 + offset, 1 + offset, t]
end

function svector_local_point(center::SVector{N, Float64}, i, nodes, node_pts) where N
    if i == 0
        pt = center
    else
        pt = node_pts[nodes[i]]
    end
    return pt::SVector{N, Float64}
end


function plot_flatten_helper(data::Vector{Tv}) where Tv<:SVector
    n = length(data)
    m = length(Tv)
    T = eltype(eltype(Tv))

    out = zeros(T, n, m)
    offset = 0
    for i in 1:n
        @. out[i, :] = data[i]
    end
    return out
end

function plot_flatten_helper(data)
    n = sum(x -> size(x, 1), data)
    m = size(first(data), 2)
    T = eltype(first(data))

    out = zeros(T, n, m)
    offset = 0
    for d in data
        n_i = size(d, 1)
        @. out[(offset+1):(offset+n_i), :] = d
        offset += n_i
    end
    return out
end
