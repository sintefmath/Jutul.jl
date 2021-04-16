using MAT # .MAT file loading
export readSimGraph, MRSTSimGraph, getSparsity, getIncompMatrix
export readPatchPlot
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


function get_minimal_grid(mat_path::String)
    exported = MAT.matread(string("data/testgrids/", filename, ".mat"))
    normals = exported["G"]["faces"]["normals"]./exported["G"]["faces"]["areas"];

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
