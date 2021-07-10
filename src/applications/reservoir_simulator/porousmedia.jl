export compute_half_face_trans, compute_face_trans
using LinearAlgebra

function compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, N)
    nf = size(N, 2)
    nhf = 2*nf
    dim = size(cell_centroids, 1)

    T_hf = similar(cell_centroids, nhf)
    faces, facePos = get_facepos(N)
    nc = length(facePos)-1
    # Sanity check
    @assert(dim == 2 || dim == 3)
    # Check cell centroids
    @assert(size(cell_centroids, 1) == dim)
    @assert(size(cell_centroids, 2) == nc)
    # Check face centroids
    @assert(size(face_centroids, 1) == dim)
    @assert(size(face_centroids, 2) == nf)
    # Check normals
    @assert(size(face_normals, 1) == dim)
    @assert(size(face_normals, 2) == nf)
    # Check areas
    @assert(length(face_areas) == nf)
    # Check perm
    @assert(size(perm, 2) == nc)
    # Check N, just in case
    @assert(size(N, 2) == nf)
    @threads for cell = 1:nc
        for fpos = facePos[cell]:(facePos[cell+1]-1)
            face = faces[fpos]

            A = face_areas[face]
            K = expand_perm(perm[:, cell], dim)
            C = face_centroids[:, face] - cell_centroids[:, cell]
            Nn = face_normals[:, face]
            if N[2, face] == cell
                Nn = -Nn
            end
            T_hf[fpos] = compute_half_face_trans(A, K, C, Nn)
        end
    end
    return T_hf
end

function expand_perm(K, dim)
    if length(K) == dim
        return diagm(K)
    else
        @assert(length(K) == 1)
        return K[1]
    end
end

function compute_half_face_trans(A, K, C, N)
    return A*(dot(K*C, N))/dot(C, C)
end

function compute_face_trans(T_hf, N)
    faces, facePos = get_facepos(N)
    nf = size(N, 2)
    T = zeros(nf)
    for i in eachindex(faces)
        T[faces[i]] += 1/T_hf[i]
    end
    T = 1 ./T
    return T
    # (N[:, 2] .> 0) .& (N[:, 1] .> 0)
end
