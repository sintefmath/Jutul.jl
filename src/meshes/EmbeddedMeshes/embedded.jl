Jutul.dim(mesh::EmbeddedMesh) = Jutul.dim(mesh.unstructured_mesh)
Jutul.number_of_cells(mesh::EmbeddedMesh) = Jutul.number_of_cells(mesh.unstructured_mesh)
Jutul.number_of_faces(mesh::EmbeddedMesh) = Jutul.number_of_faces(mesh.unstructured_mesh)
Jutul.number_of_boundary_faces(mesh::EmbeddedMesh) = Jutul.number_of_boundary_faces(mesh.unstructured_mesh)
Jutul.count_entities(mesh::EmbeddedMesh, ::Jutul.Cells) = Jutul.count_entities(mesh.unstructured_mesh, Jutul.Cells())
Jutul.count_entities(mesh::EmbeddedMesh, ::Jutul.Faces) = Jutul.count_entities(mesh.unstructured_mesh, Jutul.Faces())
Jutul.count_entities(mesh::EmbeddedMesh, ::Jutul.BoundaryFaces) = Jutul.count_entities(mesh.unstructured_mesh, Jutul.BoundaryFaces())
Jutul.get_neighborship(mesh::EmbeddedMesh; internal = true) = Jutul.get_neighborship(mesh.unstructured_mesh; internal = internal)
Jutul.mesh_entity_tags(mesh::EmbeddedMesh) = Jutul.mesh_entity_tags(mesh.unstructured_mesh)

function Jutul.cell_dims(g::EmbeddedMesh, pos)
    g = g.unstructured_mesh
    index = cell_index(g, pos)
    # Pick the nodes
    T = eltype(g.node_points)
    minv = Inf .+ zero(T)
    maxv = -Inf .+ zero(T)
    for face_set in [g.faces, g.boundary_faces]
        for face in face_set.cells_to_faces[index]
            for node in face_set.faces_to_nodes[face]
                pt = g.node_points[node]
                minv = min.(pt, minv)
                maxv = max.(pt, maxv)
            end
        end
    end
    Δ = maxv - minv
    # @assert all(x -> x > 0, Δ) "Cell dimensions were zero? Computed $Δ for cell $index."
    return Tuple(Δ)
end