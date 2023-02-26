function compute_half_face_trans(g, perm)
    compute_half_face_trans(g.cell_centroids, g.face_centroids, g.normals, g.areas, perm, g.neighbors)
end

function compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, N)
    nf = size(N, 2)
    nhf = 2*nf
    dim = size(cell_centroids, 1)

    T_hf = similar(cell_centroids, nhf)
    faces, facePos = get_facepos(N)
    nc = length(facePos)-1
    if isa(perm, AbstractFloat)
        perm = repeat([perm], 1, nc)
    else
        perm::AbstractVecOrMat
        if perm isa AbstractVector
            perm = reshape(perm, 1, :)
        end
    end

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
    Threads.@threads for cell = 1:nc
        @inbounds for fpos = facePos[cell]:(facePos[cell+1]-1)
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

function expand_perm(K, dim; full = false)
    n = length(K)
    if n == dim
        K_e = diagm(K)
    elseif n == 1
        K_e = first(K)
        if full
            # Expand to matrix
            K_e = zeros(dim, dim) + I*K_e
        end
    else
        if dim == 2
            @assert n == 3 "Two-dimensional grids require 1/2/3 permeability entries per cell (was $n)"
            K_e = [K[1] K[2]; K[2] K[3]]
        else
            @assert n == 6 "Three-dimensional grids require 1/3/6 permeability entries per cell (was $n)"
            K_e =  [K[1] K[2] K[3];
                    K[2] K[4] K[5];
                    K[3] K[5] K[6]]
        end
    end
    return K_e
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
end

function compute_face_trans(g::JutulMesh, perm)
    geo = tpfv_geometry(g)
    return compute_face_trans(geo, perm)
end

export compute_face_trans
function compute_face_trans(geometry::JutulGeometry, permeability)
    T_hf = compute_half_face_trans(geometry, permeability)
    return compute_face_trans(T_hf, geometry.neighbors)
end

export compute_face_gdz

function compute_face_gdz(g::JutulMesh; kwarg...)
    geo = tpfv_geometry(g)
    N = geo.neighbors
    if dim(geo) == 3
        z = vec(geo.cell_centroids[3, :])
    else
        z = zeros(size(geo.cell_centroids, 2))
    end
    return compute_face_gdz(N, z; kwarg...)
end

function compute_face_gdz(N, z; g = gravity_constant)
    nf = size(N, 2)
    gdz = similar(z, nf)
    for i in 1:nf
        l = N[1, i]
        r = N[2, i]
        gdz[i] = -g*(z[l] - z[r])
    end
    return gdz
end
