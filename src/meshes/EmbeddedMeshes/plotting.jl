function Jutul.triangulate_mesh(mesh::EmbeddedMesh; outer = false, flatten = true)
    m = mesh.unstructured_mesh
    D = Jutul.dim(mesh)
    
    val_cells = Jutul.Cells()
    nc = Jutul.number_of_cells(mesh)
    cell_centroids = [Jutul.compute_centroid_and_measure(mesh, val_cells, i)[1] for i in 1:nc]

    pts = Vector{SVector{D, Float64}}()
    tri = Vector{SVector{3, Int64}}()
    cell_index = Vector{Int64}()
    face_index = Vector{Int64}()

    offset = 0
    # Preallocate roughly
    dest = (cell_index, face_index, pts, tri)
    
    add_points!(e, e_def, offset, face_offset) = triangulate_and_add_faces!(dest, mesh, e, e_def, cell_centroids, offset = offset, face_offset = face_offset)
    
    if !outer
        offset = add_points!(Jutul.Faces(), m.faces, offset, 0)
    end
    face_offset = Jutul.number_of_faces(m)
    offset = add_points!(Jutul.BoundaryFaces(), m.boundary_faces, offset, face_offset)

    if flatten
        pts = flattened_data(pts)
        tri = flattened_data(tri)
    end
    
    cell_buffer = zeros(length(cell_index))
    face_buffer = zeros(length(face_index))

    mapper = (
                Cells = (cell_data) -> mesh_data_to_tris_local!(cell_buffer, cell_data, cell_index)::Vector{Float64},
                Faces = (face_data) -> mesh_data_to_tris_local!(face_buffer, face_data, face_index)::Vector{Float64},
                indices = (Cells = cell_index, Faces = face_index)
            )
    return (points = pts, triangulation = tri, mapper = mapper)
end

function triangulate_and_add_faces!(dest, m::EmbeddedMesh, e, faces, cell_centroids; offset = 0, face_offset = 0)
    node_pts = m.unstructured_mesh.node_points
    for f in 1:Jutul.count_entities(m, e)
        nodes = faces.faces_to_nodes[f]
        offset = triangulate_embedded_face!(dest, f + face_offset, faces.neighbors[f], nodes, node_pts, cell_centroids, offset)
    end
    return offset
end

function triangulate_embedded_face!(dest, face, neighbors, nodes, node_pts, cell_centroids, offset)
    cell_index, face_index, pts, tri = dest
    n = length(nodes)
    
    for cell in neighbors
        if cell > 0
            # Add Cell Centroid
            push!(cell_index, cell)
            push!(face_index, face)
            push!(pts, cell_centroids[cell])
            
            # Add Nodes
            for node in nodes
                push!(cell_index, cell)
                push!(face_index, face)
                push!(pts, node_pts[node])
            end
            
            # Add Triangle: Centroid(1), Node1(2), Node2(3) -> 1-2-3
            push!(tri, SVector{3, Int}(offset+1, offset+2, offset+3))
            
            offset += (1 + n)
        end
    end
    return offset
end

function mesh_data_to_tris_local!(out::Vector{Float64}, cell_data, cell_index)
    n = length(cell_index)
    @assert length(out) == n
    for i in eachindex(cell_index)
        c = cell_index[i]
        @inbounds out[i] = cell_data[c]
    end
    return out::Vector{Float64}
end

function flattened_data(data::Vector{Tv}) where Tv<:SVector
    n = length(data)
    m = length(Tv)
    T = eltype(Tv)
    out = zeros(T, n, m)
    for i in 1:n
        out[i, :] = data[i]
    end
    return out
end