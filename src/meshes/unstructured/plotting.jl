function Jutul.triangulate_mesh(m::UnstructuredMesh{3}; is_depth = true, outer = false)
    pts = Vector{Matrix{Float64}}()
    tri = Vector{Matrix{Int64}}()
    cell_index = Vector{Int64}()
    face_index = Vector{Int64}()
    node_pts = m.node_points
    T = eltype(node_pts)
    function cyclical_tesselation(n, offset::Int)
        # Create a triangulation of face, assuming convexity
        # Each tri is two successive points on boundary connected to centroid
        out = Matrix{Int}(undef, n, 3)
        for i in 1:n
            # First column i + 1
            out[i, 1] = i + 1 + offset
            # Second column: Always the center point
            out[i, 2] = 1 + offset
            # Third column: Wrap-around to first column
            if i == 1
                out[i, 3] = n + 1 + offset
            else
                out[i, 3] = i + offset
            end
        end
        return out
    end
    offset = 0

    function add_points!(e, faces)
        for f in 1:count_entities(m, e)
            C = zero(T)
            nodes = faces.faces_to_nodes[f]
            n = length(nodes)
            for node in nodes
                C += node_pts[node]
            end
            C /= n

            local_pts = zeros(n+1, 3)
            for i = 1:3
                xyz = C[i]
                if i == 3 && is_depth
                    xyz = -xyz
                end
                local_pts[1, i] = xyz
            end
            for (i, n) in enumerate(nodes)
                for d in 1:3
                    xyz = node_pts[n][d]
                    if d == 3 && is_depth
                        xyz = -xyz
                    end
                    local_pts[i+1, d] = xyz
                end
            end
            new_vert_count = n + 1

            for cell in faces.neighbors[f]
                if cell > 0
                    for _ in 1:new_vert_count
                        push!(cell_index, cell)
                        push!(face_index, f)
                    end
                    tri_cell = cyclical_tesselation(n, offset)::Matrix{Int}
                    push!(pts, local_pts)
                    push!(tri, tri_cell)

                    offset = offset + new_vert_count
                end
            end
        end
    end

    if !outer
        add_points!(Faces(), m.faces)
    end
    add_points!(BoundaryFaces(), m.boundary_faces)

    pts = plot_flatten_helper(pts)
    tri = plot_flatten_helper(tri)

    mapper = (
                Cells = (cell_data) -> cell_data[cell_index],
                Faces = (face_data) -> face_data[face_index],
                indices = (Cells = cell_index, Faces = face_index)
              )
    return (points = pts, triangulation = tri, mapper = mapper)
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
