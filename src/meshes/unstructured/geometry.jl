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

                M = @SMatrix [
                    l_node[1] l_node[2] l_node[3] 1;
                    r_node[1] r_node[2] r_node[3] 1;
                    c_node[1] c_node[2] c_node[3] 1;
                    c_node_face[1] c_node_face[2] c_node_face[3] 1;
                ]
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

function tpfv_geometry(G::UnstructuredMesh{D}) where D
    nc = number_of_cells(G)
    nf = number_of_faces(G)
    # Cell geometry
    cell_centroids = zeros(D, nc)
    volumes = zeros(nc)
    for i in 1:nc
        c, volumes[i] = compute_centroid_and_measure(G, Cells(), i)
        for d in 1:D
            cell_centroids[d, i] = c[d]
        end
    end

    # Face geometry
    nf = number_of_faces(G)
    N = zeros(Int, 2, nf)
    for (i, lr) in enumerate(G.faces.neighbors)
        N[1, i] = lr[1]
        N[2, i] = lr[2]
    end
    pts = G.node_points
    nf = number_of_faces(G)

    function face_geometry(e, faces)
        nf = count_entities(G, e)
        areas = zeros(nf)
        normals = zeros(D, nf)
        centroids = zeros(D, nf)
        for f in 1:nf
            c, a = compute_centroid_and_measure(G, e, f)
            for d in 1:D
                centroids[d, f] = c[d]
            end
            areas[f] = a
            # Assume correct order for normal
            nodes = faces.faces_to_nodes[f]
            a = pts[nodes[1]]
            b = pts[nodes[2]]
            c = pts[nodes[3]]

            normal = cross(c - b, a - b)
            normal /= norm(normal, 2)

            for d in 1:D
                normals[d, f] = normal[d]
            end
        end
        return (areas, normals, centroids)
    end

    face_areas, face_normals, face_centroids = face_geometry(Faces(), G.faces)
    boundary_areas, boundary_normals, boundary_centroids = face_geometry(BoundaryFaces(), G.boundary_faces)

    return TwoPointFiniteVolumeGeometry(
        N,
        face_areas,
        volumes,
        face_normals,
        cell_centroids,
        face_centroids,
        boundary_areas = boundary_areas,
        boundary_normals = boundary_normals,
        boundary_centroids = boundary_centroids,
        boundary_neighbors = G.boundary_faces.neighbors
    )
end

