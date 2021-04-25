using MAT # .MAT file loading
export readSimGraph, getSparsity, getIncompMatrix
export readPatchPlot
export get_minimal_tpfa_grid_from_mrst, plot_mrstdata
using SparseArrays # Sparse pattern
using Makie

struct MRSTPlotData
    faces::Array
    vertices::Array
    data::Vector
end


function get_minimal_tpfa_grid_from_mrst(name::String; relative_path=true, perm = nothing, poro = nothing, volumes = nothing, extraout = false)
    if relative_path
        fn = string("data/testgrids/", name, ".mat")
    else
        fn = name
    end
    @debug "Reading MAT file $fn..."
    exported = MAT.matread(fn)
    @debug "File read complete. Unpacking data..."

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

    @debug "Data unpack complete. Starting transmissibility calculations."
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
        @inbounds for fpos = facePos[cell]:(facePos[cell+1]-1)
            face = faces[fpos]
            if N[1, face] == cell
                other = N[2, face]
            else
                other = N[1, face]
            end
            if size(cell_centroids, 1) == 3
                dz = cell_centroids[3, cell] - cell_centroids[3, other]
            else
                dz = 0
            end
            faceData[fpos] = t(T[face], dz, cell, other)
        end
    end
    @debug "Setting up TPFA grid."
    sg = MinimalTPFAGrid(faceData, pv)
    if extraout
        return (sg, exported)
    else
        return sg
    end
end

function read_patch_plot(filename::String)
    vars = MAT.matread(filename)
    f = vars["faces"];
    v = vars["vertices"];
    d = vec(vars["data"])
    MRSTPlotData(f, v, d)
end

function plot_mrstdata(mrst_grid, data)
    if any([t == "cartGrid" for t in mrst_grid["type"]])
        cartDims = Int64.(mrst_grid["cartDims"])
        if mrst_grid["griddim"] == 2 || cartDims[end] == 1
            heatmap(reshape(data, cartDims[1], cartDims[2]))
        else
            println("3D Cartesian plot not implemented.")
        end
    else
        println("Non-Cartesian plot not implemented.")
    end
end
