function Jutul.triangulate_mesh(m::UnstructuredMesh{3}; is_depth = true, outer = false, flatten = true)
    N = 3
    pts = Vector{SVector{N, Float64}}()
    tri = Vector{SVector{N, Int64}}()
    cell_index = Vector{Int64}()
    face_index = Vector{Int64}()

    offset = 0

    dest = (cell_index, face_index, pts, tri)
    for d in dest
        # Assume hexahedral, 6 faces per cell, triangulated into 4 parts each
        sizehint!(d, 24*number_of_cells(m))
    end

    add_points!(e, e_def, offset) = triangulate_and_add_faces!(dest, m, e, e_def, offset = offset)
    if !outer
        offset = add_points!(Faces(), m.faces, offset)
    end
    offset = add_points!(BoundaryFaces(), m.boundary_faces, offset)

    if is_depth
        u = SVector{N, Float64}(1.0, 1.0, -1.0)
        for i in eachindex(pts)
            pts[i] = pts[i] .* u
        end
    end
    if flatten
        pts = plot_flatten_helper(pts)
        tri = plot_flatten_helper(tri)
    end
    mapper = (
                Cells = (cell_data) -> cell_data[cell_index],
                Faces = (face_data) -> face_data[face_index],
                indices = (Cells = cell_index, Faces = face_index)
              )
    return (points = pts, triangulation = tri, mapper = mapper)
end

function triangulate_and_add_faces!(dest, m, e, faces; offset = 0)
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
        offset = triangulate_and_add_faces!(dest, f, faces.neighbors[f], C, nodes, node_pts, n; offset = offset)
    end
    return offset
end


function triangulate_and_add_faces!(dest, face, neighbors, C, nodes, node_pts, n; offset = 0)
    cell_index, face_index, pts, tri = dest
    if n == 4
        # TODO: Could add a 3 mesh specialization here
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
