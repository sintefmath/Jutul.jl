
export get_cc_grid, get_boundary, get_tensorprod, get_simple_elyte_model
export exported_model_to_domain, get_ref_states, get_simple_elyte_sim

function get_boundary(name)
    fn = string(dirname(pathof(Jutul)), "/../data/testgrids/", name, "_T.mat")
    exported = MAT.matread(fn)

    exported
    bccells = copy((exported["bccells"]))
    T = copy((exported["T"]))

    bccells = Int64.(bccells)

    return (bccells[:, 1], T[:, 1])
end

function get_tensorprod(name="square_current_collector")
    fn = string(dirname(pathof(Jutul)), "/../data/testgrids/", name, "_P.mat")
    exported = MAT.matread(fn)

    exported
    P = copy((exported["P"]))
    S = copy((exported["S"]))

    return P, S
end

function get_cc_grid(
    ;name="square_current_collector", extraout = false, bc=[], b_T_hf=[], tensor_map=false
    )
    fn = string(dirname(pathof(Jutul)), "/../data/testgrids/", name, ".mat")
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
    T = compute_face_trans(T_hf, N) * 2

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

function exported_model_to_domain(exported; bc=[], b_T_hf=[], tensor_map=false, vf=[])
    N = exported["G"]["faces"]["neighbors"]
    N = Int64.(N)
    internal_faces = (N[:, 2] .> 0) .& (N[:, 1] .> 0)
    N = copy(N[internal_faces, :]')
        
    face_areas = vec(exported["G"]["faces"]["areas"][internal_faces])
    face_normals = exported["G"]["faces"]["normals"][internal_faces, :]./face_areas
    face_normals = copy(face_normals')
    if length(exported["G"]["cells"]["volumes"])==1
        volumes = exported["G"]["cells"]["volumes"]
        volumes = Vector{Float64}(undef,1)
        volumes[1] = exported["G"]["cells"]["volumes"]
    else
        volumes = vec(exported["G"]["cells"]["volumes"])
    end
    P = exported["operators"]["cellFluxOp"]["P"]
    S = exported["operators"]["cellFluxOp"]["S"]
    G = MinimalECTPFAGrid(volumes, N, bc, b_T_hf, P, S, vf)

    T = exported["operators"]["T"].*2.0*-1.0
    flow = TPFlow(G, T, tensor_map=tensor_map)
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

function get_simple_elyte_model(name="modelElectrolyte")
    fn = string(dirname(pathof(Jutul)), "/../data/models/", name, ".mat")
    exported = MAT.matread(fn)
    ex_model = exported["model"]

    boundary = ex_model["bcstruct"]["dirichlet"]

    b_faces = Int64.(boundary["faces"])
    T_all = ex_model["operators"]["T_all"]
    N_all = Int64.(ex_model["G"]["faces"]["neighbors"])
    isboundary = (N_all[b_faces, 1].==0) .| (N_all[b_faces, 2].==0)
    @assert all(isboundary)
    bc_cells = N_all[b_faces, 1] + N_all[b_faces, 2]
    b_T_hf   = T_all[b_faces]

    domain = exported_model_to_domain(ex_model, bc=bc_cells, b_T_hf=b_T_hf)

    sys = SimpleElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryC] = BoundaryPotential{C}()

    return model, exported
end

function get_simple_elyte_sim(model, exported)
    boundary = exported["model"]["bcstruct"]["dirichlet"]
    b_phi = boundary["phi"][:, 1]
    b_c = boundary["conc"][:, 1]
    init_states = exported["state0"]
    init = Dict(
        :Phi            => init_states["phi"][:, 1],
        :C              => init_states["cs"][1][:, 1],
        :T              => init_states["T"][:, 1],
        :BoundaryPhi    => b_phi,
        :BoundaryC      => b_c,
        )

    state0 = setup_state(model, init)

    parameters = setup_parameters(model)
    parameters[:tolerances][:default] = 1e-10
    t1, t2 = exported["model"]["sp"]["t"]
    z1, z2 = exported["model"]["sp"]["z"]
    tDivz_eff = (t1/z1 + t2/z2)
    parameters[:t] = tDivz_eff
    parameters[:z] = 1
 
    sim = Simulator(model, state0=state0, parameters=parameters)

    return sim
end