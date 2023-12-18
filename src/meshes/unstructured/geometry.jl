function compute_centroid_and_measure(G::UnstructuredMesh{3}, ::Faces, i)
    nodes = G.faces.faces_to_nodes[i]
    pts = G.node_points
    return face_centroid_and_measure(nodes, pts)
end

function compute_centroid_and_measure(G::UnstructuredMesh{3}, ::BoundaryFaces, i)
    nodes = G.boundary_faces.faces_to_nodes[i]
    pts = G.node_points
    return face_centroid_and_measure(nodes, pts)
end

function face_centroid_and_measure(nodes, pts)
    T = eltype(pts)
    c_node = zero(T)
    for n in nodes
        c_node += pts[n]
    end
    c_node /= length(nodes)

    area = 0.0
    centroid = zero(T)
    for i in 1:(length(nodes))
        if i == 1
            l = nodes[end]
        else
            l = nodes[i-1]
        end
        r = nodes[i]
        l_node = pts[l]
        r_node = pts[r]
        B = l_node - c_node
        A = r_node - c_node

        c = (1/3)*(l_node + r_node + c_node)
        tri_area = 0.5*norm(cross(A, B), 2)
        area += tri_area
        centroid += c*tri_area
    end
    return (centroid./area, area)
end

function compute_centroid_and_measure(G::UnstructuredMesh{3}, ::Cells, i)
    pts = G.node_points
    T = eltype(pts)
    c_node = zero(T)
    n = 0

    function sum_points(faces, n, c_node, i)
        for face in faces.cells_to_faces[i]
            for node in faces.faces_to_nodes[face]
                c_node += pts[node]
                n += 1
            end
        end
        return (c_node, n)
    end

    c_node, n = sum_points(G.faces, n, c_node, i)
    c_node, n = sum_points(G.boundary_faces, n, c_node, i)
    c_node /= n

    vol = 0.0
    centroid = zero(T)

    function sum_volumes(faces, centroid, vol, i)
        for face in faces.cells_to_faces[i]
            nodes = faces.faces_to_nodes[face]
            # Compute center point (not centroid) for face
            c_node_face = zero(T)
            for node in nodes
                c_node_face += pts[node]
            end
            c_node_face /= length(nodes)
            # Then create tets and compute volume
            for i in eachindex(nodes)
                if i == 1
                    l = nodes[end]
                else
                    l = nodes[i-1]
                end
                r = nodes[i]
                l_node = pts[l]
                r_node = pts[r]

                M = SMatrix{4, 4, Float64, 16}(
                    l_node[1], r_node[1], c_node[1], c_node_face[1],
                    l_node[2], r_node[2], c_node[2], c_node_face[2],
                    l_node[3], r_node[3], c_node[3], c_node_face[3],
                    1.0, 1.0, 1.0, 1.0
                )
                local_volume = (1/6)*abs(det(M))
                local_centroid = (1/4)*(l_node + r_node + c_node_face + c_node)

                vol += local_volume
                centroid += local_centroid*local_volume
            end
        end
        return (centroid, vol)
    end
    centroid, vol = sum_volumes(G.faces, centroid, vol, i)
    centroid, vol = sum_volumes(G.boundary_faces, centroid, vol, i)
    return (centroid./vol, vol)
end

