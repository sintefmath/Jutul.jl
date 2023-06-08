export compute_face_trans, compute_half_face_trans, compute_boundary_trans

function compute_half_face_trans(d::DataDomain, k::Symbol = :permeability)
    @assert haskey(d, k, Cells()) "$k must be defined in Cells."
    return compute_half_face_trans(d, d[k])
end

"""
    compute_half_face_trans(g::DataDomain, perm)

Compute half-face trans for the interior faces. The input `perm` can either be
the symbol of some data defined on `Cells()`, a vector of numbers for each cell
or a matrix with number of columns equal to the number of cells.
"""
function compute_half_face_trans(g::DataDomain, perm)
    compute_half_face_trans(g[:cell_centroids], g[:face_centroids], g[:normals], g[:areas], perm, g[:neighbors])
end

function compute_half_face_trans(g::TwoPointFiniteVolumeGeometry, perm)
    compute_half_face_trans(g.cell_centroids, g.face_centroids, g.normals, g.areas, perm, g.neighbors)
end

function compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, N)
    nc = size(cell_centroids, 2)
    @assert size(N) == (2, length(face_areas))
    faces, facepos = get_facepos(N, nc)
    facesigns = similar(faces)
    for c in 1:nc
        for ix in facepos[c]:(facepos[c+1]-1)
            f = faces[ix]
            if N[2, f] == c
                facesigns[ix] = -1
            else
                facesigns[ix] = 1
            end
        end
    end
    compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, faces, facepos, facesigns)
end


function compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, faces, facepos, facesigns)
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
    @assert(dim == 2 || dim == 3 || dim == 1)
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
    for cell = 1:nc
        for fpos = facepos[cell]:(facepos[cell+1]-1)
            face = faces[fpos]
            A = face_areas[face]
            K = expand_perm(perm[:, cell], dim)
            C = face_centroids[:, face] - cell_centroids[:, cell]
            Nn = facesigns[fpos]*face_normals[:, face]
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

function compute_face_trans(geometry::JutulGeometry, permeability)
    T_hf = compute_half_face_trans(geometry, permeability)
    return compute_face_trans(T_hf, geometry.neighbors)
end

"""
compute_face_trans(g::DataDomain, perm)

Compute face trans for the interior faces. The input `perm` can either be
the symbol of some data defined on `Cells()`, a vector of numbers for each cell
or a matrix with number of columns equal to the number of cells.
"""
function compute_face_trans(d::DataDomain, arg...)
    T_hf = compute_half_face_trans(d, arg...)
    return compute_face_trans(T_hf, d[:neighbors])
end

function compute_boundary_trans(d::DataDomain, k::Symbol = :permeability)
    @assert haskey(d, k, Cells()) "$k must be defined in Cells."
    return compute_boundary_trans(d, d[k])
end

"""
    compute_boundary_trans(d::DataDomain, perm)

Compute the boundary half face transmissibilities for perm. The input `perm` can
either be the symbol of some data defined on `Cells()`, a vector of numbers
for each cell or a matrix with number of columns equal to the number of
cells.
"""
function compute_boundary_trans(d::DataDomain, perm)
    @assert hasentity(d, BoundaryFaces()) "Domain must have BoundaryFaces() to compute boundary transmissibilities"
    face_areas = d[:boundary_areas]
    cells = d[:boundary_neighbors]
    cell_centroids = d[:cell_centroids][:, cells]
    face_centroids = d[:boundary_centroids]
    face_normals = d[:boundary_normals]
    if perm isa AbstractVector
        perm = perm[cells]
    else
        perm = perm[:, cells]
    end
    nf = length(face_areas)
    nc = length(cells)
    @assert nf == nc "$nf != $nc"
    faces = collect(1:nf)
    facepos = collect(1:(nc+1))
    facesigns = ones(nf)
    return compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, faces, facepos, facesigns)
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
        gdz[i] = -g*(z[r] - z[l])
    end
    return gdz
end
