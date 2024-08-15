function compute_centroid_and_measure(G::UnstructuredMesh, ::Faces, i)
    nodes = G.faces.faces_to_nodes[i]
    pts = G.node_points
    return face_centroid_and_measure(nodes, pts)
end

function compute_centroid_and_measure(G::UnstructuredMesh, ::BoundaryFaces, i)
    nodes = G.boundary_faces.faces_to_nodes[i]
    pts = G.node_points
    return face_centroid_and_measure(nodes, pts)
end

function face_centroid_and_measure(nodes, pts::Vector{SVector{3, Num}}) where {Num}
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

function face_centroid_and_measure(nodes, pts::Vector{SVector{2, Num}}) where {Num}
    @assert length(nodes) == 2
    l, r = nodes
    centroid = (pts[l] + pts[r])/2.0
    area = norm(pts[l] - pts[r], 2)
    return (centroid, area)
end

function compute_centroid_and_measure(G::UnstructuredMesh, ::Cells, i)
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

    centroid, vol = sum_centroid_volumes_helper(pts, c_node, G.faces, centroid, vol, i)
    centroid, vol = sum_centroid_volumes_helper(pts, c_node, G.boundary_faces, centroid, vol, i)
    return (centroid./vol, vol)
end

function sum_centroid_volumes_helper(pts::Vector{SVector{N, E}}, c_node::SVector{N, E}, faces, centroid::SVector{N, E}, vol, i) where {N, E}
    T = SVector{N, E}
    for face in faces.cells_to_faces[i]
        nodes = faces.faces_to_nodes[face]
        # Compute center point (not centroid) for face
        c_node_face = zero(T)
        for node in nodes
            c_node_face += pts[node]
        end
        c_node_face /= length(nodes)
        if N == 3
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
                if N == 3
                    M = SMatrix{4, 4, Float64, 16}(
                        l_node[1], r_node[1], c_node[1], c_node_face[1],
                        l_node[2], r_node[2], c_node[2], c_node_face[2],
                        l_node[3], r_node[3], c_node[3], c_node_face[3],
                        1.0, 1.0, 1.0, 1.0
                    )
                    local_volume = (1.0/6.0)*abs(det(M))
                    local_centroid = (1.0/4.0)*(l_node + r_node + c_node_face + c_node)
                else
                    A = r_node - c_node
                    B = l_node - c_node
                    local_volume = abs(cross(A, B)/4)
                    local_centroid = (l_node + r_node + c_node)/3.0
                    @assert local_volume >= 0
                end
                vol += local_volume
                centroid += local_centroid*local_volume
            end
        else
            # 2D is much simpler (area = volume in Jutul)
            @assert length(nodes) == 2
            l, r = nodes
            l_node = pts[l]
            r_node = pts[r]
            A = l_node - c_node
            B = r_node - c_node
            local_volume = abs(cross(A, B)/2.0)
            local_centroid = (l_node + r_node + c_node)/3.0
            vol += local_volume
            centroid += local_centroid*local_volume
        end
    end
    return (centroid, vol)
end

function face_normal(G::UnstructuredMesh{3}, f, e = Faces())
    get_nodes(::Faces) = G.faces
    get_nodes(::BoundaryFaces) = G.boundary_faces
    nodes = get_nodes(e).faces_to_nodes[f]
    pts = G.node_points
    n = length(nodes)
    # If the geometry is well defined it would be sufficient to take the first
    # triplet and use that to generate the normals. We assume it isn't and
    # create a weighted sum where each weight corresponds to the areal between
    # the triplets.
    normal = zero(eltype(pts))
    for i in 1:n
        if i == 1
            a = pts[nodes[n]]
        else
            a = pts[nodes[i-1]]
        end
        b = pts[nodes[i]]
        if i == n
            c = pts[nodes[1]]
        else
            c = pts[nodes[i+1]]
        end
        normal += cross(c - b, a - b)
    end
    normal /= norm(normal, 2)
    return normal
end

function face_normal(G::UnstructuredMesh{2}, f, e = Faces())
    get_nodes(::Faces) = G.faces
    get_nodes(::BoundaryFaces) = G.boundary_faces
    nodes = get_nodes(e).faces_to_nodes[f]
    pts = G.node_points
    n = length(nodes)
    @assert n == 2
    T = eltype(pts)
    l, r = nodes
    pt_l = pts[l]
    pt_r = pts[r]

    v = pt_r - pt_l
    normal = T(v[2], -v[1])

    return normal/norm(normal, 2)
end

