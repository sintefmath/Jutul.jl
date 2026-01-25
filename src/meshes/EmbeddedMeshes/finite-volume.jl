function Jutul.compute_half_face_trans(mesh::EmbeddedMesh, cell_centroids, face_centroids, face_areas, perm, faces, facepos)
    nf = length(face_areas)
    dim = size(cell_centroids, 1)

    T_hf = zeros(eltype(face_areas), length(faces))
    nc = length(facepos)-1
    if isa(perm, AbstractFloat)
        perm = repeat([perm], 1, nc)
    else
        perm::AbstractVecOrMat
        if perm isa AbstractVector
            perm = reshape(perm, 1, :)
        end
    end

    # Sanity check
    if !(dim == 2 || dim == 3 || dim == 1)
        throw(ArgumentError("Dimension must 1, 2 or 3"))
    end
    # Check cell centroids
    cc_dim, cc_n = size(cell_centroids)
    cc_dim == dim || throw(ArgumentError("Cell centroids had $cc_dim rows but grid had $dim dimension."))
    cc_n == nc || throw(ArgumentError("Cell centroids had $cc_n columns but grid had $nc cells."))
    # Check face centroids
    fc_dim, fc_n = size(face_centroids)
    fc_dim == dim || throw(ArgumentError("Face centroids had $fc_dim rows but grid had $dim dimension."))
    fc_n == nf || throw(ArgumentError("Face centroids had $fc_n columns but grid had $nf faces."))
    # Check normals
    fc_n == nf || throw(ArgumentError("Face normals had $normal_n columns but grid had $nf faces."))
    # Check areas
    # TODO: This isn't really checking anything since we get nf from areas...
    length(face_areas) == nf || throw(ArgumentError("Face areas had $normal_n entries but grid had $nf faces."))
    # Check perm
    size(perm, 2) == nc || throw(ArgumentError("Permeability must have number of columns equal to number of cells (= $nc)."))
    vdim = Val(dim)
    cc = zeros(eltype(cell_centroids), dim)
    fc = zeros(eltype(face_centroids), dim)
    for cell = 1:nc
        cn = cell_normal(mesh, cell)
        for fpos = facepos[cell]:(facepos[cell+1]-1)
            face = faces[fpos]
            @. cc = cell_centroids[:, cell]
            @. fc = face_centroids[:, face]
            A = face_areas[face]
            C = fc - cc
            Nn = half_face_normal(mesh, face, cell, cn)
            K = Jutul.expand_perm(perm[:, cell], vdim)
            T = Jutul.half_face_trans(A, K, C, Nn)
            T = abs(T)
            T_hf[fpos] = T
        end
    end
    return T_hf
end

function cell_normal(mesh::EmbeddedMesh, c)
    # TODO: check that faces are not paralell, consider using 

    function get_face_vectors(mesh, face_u, face_v, cell)

        nodes_u = mesh.faces.faces_to_nodes[face_u]
        flip = umesh.faces.neighbors[face_u][1] != cell
        nodes_u = flip ? reverse(nodes_u) : nodes_u
        u = pts[nodes_u[2]] - pts[nodes_u[1]]

        nodes_v = mesh.faces.faces_to_nodes[face_v]
        nodes_v = setdiff(nodes_v, nodes_u)
        
        if !isempty(nodes_v)
            v = pts[nodes_v[1]] - pts[nodes_u[1]] 
        else
            v = missing
        end

        return u,v

    end

    umesh = mesh.unstructured_mesh
    faces = umesh.faces.cells_to_faces[c]
    @assert length(faces) >= 2

    pts = umesh.node_points
    num_faces = length(faces)
    for i = 1:num_faces-1
        u,v = get_face_vectors(umesh, faces[i], faces[i+1], c)
        if v === missing
            continue
        end
        normal = cross(u, v)
        if norm(normal, 2) > 0
            return normal/norm(normal, 2)
        end
    end

end

function half_face_normal(mesh::EmbeddedMesh, face, cell, cn)

    umesh = mesh.unstructured_mesh
    nodes = umesh.faces.faces_to_nodes[face]
    flip = umesh.faces.neighbors[face][1] != cell
    nodes = flip ? reverse(nodes) : nodes

    pts = umesh.node_points
    vec = pts[nodes[2]] - pts[nodes[1]]

    normal = cross(vec, cn)
    normal /= norm(normal, 2)

    return normal

end

function compute_face_trans_dfm(T_hf, N, intersections)
    T = Jutul.compute_face_trans(T_hf, N)
    # Adjust transmissibilities for intersection faces
    faces, face_pos = get_facepos(N)
    nf = diff(face_pos)
    cells = vcat([fill(i, nf[i]) for i in 1:length(nf)]...)
    for ix_cells in intersections
        # Find all faces belonging to this intersection
        ix_faces = Int[]
        for (face, n) in enumerate(eachcol(N))
            if all(c in ix_cells for c in n)
                push!(ix_faces, face)
            end
        end
        den = 0.0
        T_ix = zeros(Float64, length(ix_faces))
        for (fno, f) in enumerate(ix_faces)
            ci = N[1, f]
            cj = N[2, f]
            ii = faces .== f .&& cells .== ci
            jj = faces .== f .&& cells .== cj
            @assert sum(ii) == 1 && sum(jj) == 1
            ii, jj = findfirst(ii), findfirst(jj)
            T_ix[fno] = T_hf[ii].*T_hf[jj]
            den += T_hf[ii] + T_hf[jj]
        end
        T_ix ./= den
        for (f, T_val) in zip(ix_faces, T_ix)
            T[f] = T_val
        end
    end
    return T
end