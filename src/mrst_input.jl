using MAT # .MAT file loading
export readSimGraph, MRSTSimGraph, getSparsity, getIncompMatrix
export readPatchPlot
export get_minimal_tpfa_grid_from_mrst
using SparseArrays # Sparse pattern

struct MRSTSimGraph
    faces::Vector
    self::Vector # Self cell for each half face
    cells::Vector # Other cell for each half face
    facePos::Vector
    faceSign::Vector
    T::Vector
    HalfFaceData::Vector
    N::Matrix
    pv::Vector
    ncells
    nfaces
end

struct MRSTPlotData
    faces::Array
    vertices::Array
    data::Vector
end


function get_minimal_tpfa_grid_from_mrst(name::String; relative_path=true, perm = nothing, poro = nothing, volumes = nothing)
    if relative_path
        fn = string("data/testgrids/", name, ".mat")
    else
        fn = name
    end
    exported = MAT.matread(fn)
    N = exported["G"]["faces"]["neighbors"]
    N = Int64.(N)
    
    internal_faces = (N[:, 2] .> 0) .& (N[:, 1] .> 0)
    N = copy(N[internal_faces, :]')
    
    # get_cell_faces(N)
    # get_cell_neighbors(N)
    
    # Cells
    cell_centroids = copy((exported["G"]["cells"]["centroids"])')
    # Faces
    face_centroids = copy((exported["G"]["faces"]["centroids"][internal_faces, :])')
    face_areas = vec(exported["G"]["faces"]["areas"][internal_faces])
    face_normals = exported["G"]["faces"]["normals"][internal_faces, :]./face_areas
    face_normals = copy(face_normals')
    if isnothing(perm)
        perm = copy((exported["rock"]["perm"])')
    end

    # Deal with cell data
    if isnothing(poro)
        poro = vec(exported["rock"]["poro"])
    end
    if isnothing(volumes)
        volumes = vec(exported["G"]["cells"]["volumes"])
    end
    pv = poro.*volumes
    nc = length(pv)

    # Deal with face data
    T_hf = compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, N)
    T = compute_face_trans(T_hf, N)

    nhf = length(T_hf)
    faces, facePos = get_facepos(N)

    float_type = typeof(T[1])
    index_type = typeof(N[1])
    t = TPFAHalfFaceData{float_type, index_type}
    faceData = Vector{t}(undef, nhf)

    for cell = 1:nc
        for fpos = facePos[cell]:(facePos[cell+1]-1)
            face = faces[fpos]
            if N[1, face] == cell
                other = N[2, face]
            else
                other = N[1, face]
            end
            dz = cell_centroids[3, cell] - cell_centroids[3, other]
            faceData[fpos] = t(T[face], dz, cell, other)
        end
    end
    sg = MinimalTPFAGrid(faceData, pv)
end


function read_sim_graph(filename::String)
    vars = MAT.matread(filename)
    f = vec(vars["faces"]);
    c = vec(vars["cells"]);
    s = vec(vars["self"]);
    fp = vec(vars["facePos"])
    fs = vec(vars["faceSign"])
    t = vec(vars["trans"])
    thf = t[f]
    N = vars["N"]
    pv = vec(vars["pv"])
    nc = length(pv)
    nf = size(N, 2)

    nhf = 2*nf
    faceData = Vector{HalfFaceData{typeof(thf[1]), typeof(s[1])}}(undef, nhf)
    for i = 1:nhf
        faceData[i] = HalfFaceData(thf[i], s[i], c[i])
    end
    sg = MRSTSimGraph(f, s, c, fp, fs, t, faceData, N, pv, nc, nf)
end

function read_patch_plot(filename::String)
    vars = MAT.matread(filename)
    f = vars["faces"];
    v = vars["vertices"];
    d = vec(vars["data"])
    MRSTPlotData(f, v, d)
end
