export compute_face_trans, compute_half_face_trans, compute_boundary_trans

function compute_half_face_trans(d::DataDomain, k::Symbol = :permeability; kwarg...)
    @assert haskey(d, k, Cells()) "$k must be defined in Cells."
    return compute_half_face_trans(d, d[k]; kwarg...)
end

"""
    compute_half_face_trans(g::DataDomain, perm)

Compute half-face trans for the interior faces. The input `perm` can either be
the symbol of some data defined on `Cells()`, a vector of numbers for each cell
or a matrix with number of columns equal to the number of cells.
"""
function compute_half_face_trans(g::DataDomain, perm; kwarg...)
    return compute_half_face_trans(
        g[:cell_centroids],
        g[:face_centroids],
        g[:normals],
        g[:areas],
        perm,
        g[:neighbors]
        ; kwarg...
    )
end

function compute_half_face_trans(g::TwoPointFiniteVolumeGeometry, perm; kwarg...)
    return compute_half_face_trans(g.cell_centroids, g.face_centroids, g.normals, g.areas, perm, g.neighbors; kwarg...)
end

function compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, N; kwarg...)
    nc = size(cell_centroids, 2)
    @assert size(N) == (2, length(face_areas))
    faces, facepos = get_facepos(N, nc)
    facesigns = get_facesigns(N, faces, facepos, nc)
    return compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, faces, facepos, facesigns; kwarg...)
end


function compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, faces, facepos, facesigns; version = :xyz, face_dir = missing)
    nf = length(face_areas)
    dim = size(cell_centroids, 1)

    nc = length(facepos)-1
    if isa(perm, Real)
        perm = repeat([perm], 1, nc)
    else
        perm::AbstractVecOrMat
        if perm isa AbstractVector
            perm = reshape(perm, 1, :)
        end
    end
    # Make sure that types match
    T_num = promote_type(eltype(face_areas), eltype(cell_centroids), eltype(face_normals), eltype(face_areas), eltype(perm))
    T_hf = zeros(T_num, length(faces))

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
    normal_dim, normal_n = size(face_normals)
    normal_dim == dim || throw(ArgumentError("Face normals had $normal_dim rows but grid had $dim dimension."))
    normal_n == nf || throw(ArgumentError("Face normals had $normal_n columns but grid had $nf faces."))
    # Check areas
    # TODO: This isn't really checking anything since we get nf from areas...
    length(face_areas) == nf || throw(ArgumentError("Face areas had $(length(face_areas)) entries but grid had $nf faces."))
    # Check perm
    size(perm, 2) == nc || throw(ArgumentError("Permeability must have number of columns equal to number of cells (= $nc)."))
    if !(version in (:xyz, :ijk))
        throw(ArgumentError("version must be :xyz or :ijk"))
    end
    if version == :ijk
        if size(perm, 1) != dim
            throw(ArgumentError("version = :ijk is only valid when perm is strictly diagonal."))
        end
        if ismissing(face_dir)
            throw(ArgumentError("version = :ijk cannot be used without also passing face_dir."))
        end
        if length(face_dir) != nf
            throw(ArgumentError("face_dir must have one entry per face."))
        end
        if maximum(face_dir) > dim
            throw(ArgumentError("face_dir entry exceeds grid dim ($dim)."))
        end
        if minimum(face_dir) < 1
            throw(ArgumentError("face_dir entry is less than 1."))
        end
    end
    vdim = Val(dim)
    is_xyz = Val(version == :xyz)
    compute_half_face_trans!(T_hf, cell_centroids, face_centroids, face_normals, face_areas, perm, faces, facepos, facesigns, vdim, is_xyz)
    return T_hf
end

function compute_half_face_trans!(T_hf, cell_centroids, face_centroids, face_normals, face_areas, perm, faces, facepos, facesigns, vdim::Val{dim}, ::Val{is_xyz}) where {dim, is_xyz}
    cc = zeros(eltype(cell_centroids), dim)
    fc = zeros(eltype(face_centroids), dim)
    Nn = zeros(eltype(face_normals), dim)

    for cell in axes(cell_centroids, 2)
        for fpos = facepos[cell]:(facepos[cell+1]-1)
            face = faces[fpos]
            sgn = facesigns[fpos]
            @. cc = cell_centroids[:, cell]
            @. fc = face_centroids[:, face]
            A = face_areas[face]
            C = fc - cc
            @. Nn = sgn*face_normals[:, face]
            if is_xyz
                perm_c = view(perm, :, cell)
                K = expand_perm(perm_c, vdim)
            else
                K = perm[face_dir[face], cell]
            end
            T = 
            T_hf[fpos] = half_face_trans(A, K, C, Nn)
        end
    end
    return T_hf
end

function expand_perm(K, dim)
    return expand_perm(K, Val(dim))
end

function expand_perm(K, ::Val{1})
    return only(K)
end


function expand_perm(K, ::Val{2})
    T = eltype(K)
    n = length(K)
    if n == 1
        K_xx = K_yy = only(K)
        K_xy = zero(T)
    elseif n == 2
        K_xx = K[1]
        K_yy = K[2]
        K_xy = zero(T)
    elseif n == 3
        K_xx = K[1]
        K_xy = K[2]
        K_yy = K[2]
    else
        error("Permeability for two-dimensional grids must have 1/2/3 entries per cell, had $n")
    end
    K_e = @SMatrix [
        K_xx K_xy;
        K_xy K_yy
        ]
    return K_e
end

function expand_perm(K, ::Val{3})
    T = eltype(K)
    n = length(K)
    K_xy = zero(T)
    K_xz = zero(T)
    K_yz = zero(T)
    if n == 1
        K_xx = K_yy = K_zz = only(K)
    elseif n == 3
        K_xx = K[1]
        K_yy = K[2]
        K_zz = K[3]
    elseif n == 6
        # First row
        K_xx = K[1]
        K_xy = K[2]
        K_yz = K[3]
        # Second row excluding symmetry
        K_yy = K[4]
        K_yz = K[5]
        # Last entry
        K_zz = K[6]
    else
        error("Permeability for three-dimensional meshes must have 1/3/6 entries per cell, had $n")
    end
    K_e =  @SMatrix[
        K_xx K_xy K_xz;
        K_xy K_yy K_yz;
        K_xz K_yz K_zz]
    return K_e
end

function half_face_trans(A, K, C, N)
    return A*(dot(K*C, N))/dot(C, C)
end

function compute_face_trans(T_hf, N)
    faces, facePos = get_facepos(N)
    @assert length(T_hf) == length(faces)
    nf = size(N, 2)
    T = zeros(eltype(T_hf), nf)
    for i in eachindex(faces)
        T[faces[i]] += 1.0/T_hf[i]
    end
    @. T = 1.0 / T
    return T
end

function compute_face_trans(g::JutulMesh, perm)
    geo = tpfv_geometry(g)
    return compute_face_trans(geo, perm)
end

function compute_face_trans(geometry::JutulGeometry, permeability; kwarg...)
    T_hf = compute_half_face_trans(geometry, permeability; kwarg...)
    return compute_face_trans(T_hf, geometry.neighbors)
end

"""
compute_face_trans(g::DataDomain, perm)

Compute face trans for the interior faces. The input `perm` can either be
the symbol of some data defined on `Cells()`, a vector of numbers for each cell
or a matrix with number of columns equal to the number of cells.
"""
function compute_face_trans(d::DataDomain, arg...; kwarg...)
    T_hf = compute_half_face_trans(d, arg...; kwarg...)
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
function compute_boundary_trans(d::DataDomain, perm; kwarg...)
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
    return compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, faces, facepos, facesigns; kwarg...)
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
