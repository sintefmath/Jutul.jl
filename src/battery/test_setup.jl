using Terv

export get_cc_grid, get_boundary, get_tensorprod
export exported_model_to_domain, get_ref_states

function get_boundary(name)
    fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, "_T.mat")
    exported = MAT.matread(fn)

    exported
    bccells = copy((exported["bccells"]))
    T = copy((exported["T"]))

    bccells = Int64.(bccells)

    return (bccells[:, 1], T[:, 1])
end

function get_tensorprod(name="square_current_collector")
    fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, "_P.mat")
    exported = MAT.matread(fn)

    exported
    P = copy((exported["P"]))
    S = copy((exported["S"]))

    return P, S
end

function get_cc_grid(
    ;name="square_current_collector", extraout = false, bc=[], b_T_hf=[], tensor_map=false
    )
    fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, ".mat")
    exported = MAT.matread(fn)

    N = exported["G"]["faces"]["neighbors"]
    N = Int64.(N)
    internal_faces = (N[:, 2] .> 0) .& (N[:, 1] .> 0)
    N = copy(N[internal_faces, :]')
        
    # Cells
    cell_centroids = copy((exported["G"]["cells"]["centroids"])')

    # Faces
    face_centroids = copy((exported["G"]["faces"]["centroids"][internal_faces, :])')
    face_areas = vec(exported["G"]["faces"]["areas"][internal_faces])
    face_normals = exported["G"]["faces"]["normals"][internal_faces, :]./face_areas
    face_normals = copy(face_normals')
    volumes = vec(exported["G"]["cells"]["volumes"])

    # Deal with face data

    # Different constants for different potential means this cannot be included in T
    one = ones(size((exported["rock"]["perm"])'))
    
    T_hf = compute_half_face_trans(
    cell_centroids, face_centroids, face_normals, face_areas, one, N
        )
    T = compute_face_trans(T_hf, N)

    # TODO: move P, S, boundary to discretization
    P, S = get_tensorprod(name)
    G = MinimalECTPFAGrid(volumes, N, bc, b_T_hf, P, S)

    flow = TPFlow(G, T; tensor_map=tensor_map)
    disc = (charge_flow = flow,)
    D = DiscretizedDomain(G, disc)

    if extraout
        return (D, exported)
    else
        return D
    end
end

function exported_model_to_domain(exported; bc=[], b_T_hf=[])
    N = exported["G"]["faces"]["neighbors"]
    N = Int64.(N)
    internal_faces = (N[:, 2] .> 0) .& (N[:, 1] .> 0)
    N = copy(N[internal_faces, :]')
        
    face_areas = vec(exported["G"]["faces"]["areas"][internal_faces])
    face_normals = exported["G"]["faces"]["normals"][internal_faces, :]./face_areas
    face_normals = copy(face_normals')
    volumes = vec(exported["G"]["cells"]["volumes"])

    P = exported["operators"]["cellFluxOp"]["P"]
    S = exported["operators"]["cellFluxOp"]["S"]
    G = MinimalECTPFAGrid(volumes, N, bc, b_T_hf, P, S)

    T = exported["operators"]["T"].*2.0
    flow = TPFlow(G, T)
    disc = (charge_flow = flow,)
    D = DiscretizedDomain(G, disc)

    return D
end

function get_ref_states(j2m, ref_states)
    m2j = Dict(value => key for (key, value) in j2m)
    rs = [ 
        Dict(m2j[k] => v[:, 1] for (k, v) in state if k in keys(m2j))
        for state in ref_states
        ]
    if :C in keys(j2m)
        [s[:C] = s[:C][1][:, 1] for s in rs]
    end
    return rs
end