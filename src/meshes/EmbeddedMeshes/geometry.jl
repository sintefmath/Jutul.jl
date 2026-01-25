Jutul.face_normal(mesh::EmbeddedMesh, f, e::Jutul.Faces) = Jutul.face_normal(mesh.unstructured_mesh, f, e)
Jutul.face_normal(mesh::EmbeddedMesh, f, e::Jutul.BoundaryFaces) = Jutul.face_normal(mesh.unstructured_mesh, f, e)

function Jutul.compute_centroid_and_measure(mesh::EmbeddedMesh, ::Jutul.Cells, i)

    umesh = mesh.unstructured_mesh
    pts = umesh.node_points
    T = eltype(pts)
    c_node = zero(T)
    n = 0

    # Collect unique faces to avoid double counting
    unique_faces = Set{Vector{Int}}()

    function collect_faces!(unique_faces, faces, cell)
        for face in faces.cells_to_faces[cell]
            nodes = faces.faces_to_nodes[face]
            push!(unique_faces, sort(nodes))
        end
    end

    collect_faces!(unique_faces, umesh.faces, i)
    collect_faces!(unique_faces, umesh.boundary_faces, i)

    for nodes in unique_faces
        for node in nodes
            c_node += pts[node]
            n += 1
        end
    end
    c_node /= n

    vol = 0.0
    centroid = zero(T)

    for nodes in unique_faces
        @assert length(nodes) == 2
        l, r = nodes
        l_node = pts[l]
        r_node = pts[r]
        A = l_node - c_node
        B = r_node - c_node
        local_volume = LinearAlgebra.norm(cross(A, B)/2.0, 2)
        local_centroid = (l_node + r_node + c_node)/3.0
        vol += local_volume
        centroid += local_centroid*local_volume
    end
    return (centroid./vol, vol)

end

function Jutul.compute_centroid_and_measure(mesh::EmbeddedMesh, ::Jutul.Faces, i)

    umesh = mesh.unstructured_mesh
    nodes = umesh.faces.faces_to_nodes[i]
    pts = umesh.node_points
    return face_centroid_and_measure(mesh, nodes, pts)

end

function Jutul.compute_centroid_and_measure(mesh::EmbeddedMesh, ::Jutul.BoundaryFaces, i)

    umesh = mesh.unstructured_mesh
    nodes = umesh.boundary_faces.faces_to_nodes[i]
    pts = umesh.node_points
    return face_centroid_and_measure(mesh, nodes, pts)

end

function face_centroid_and_measure(mesh::EmbeddedMesh, nodes, pts::Vector{SVector{3, Num}}) where {Num}

    @assert length(nodes) == 2
    l, r = nodes
    centroid = (pts[l] + pts[r])/2.0
    area = LinearAlgebra.norm(pts[l] - pts[r], 2)
    return (centroid, area)

end