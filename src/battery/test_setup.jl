using Terv

export get_test_setup_battery, get_cc_grid, get_bccc_struct

function get_test_setup_battery(name="square_current_collector")
    domain, exported = get_cc_grid(name, true)
    timesteps = [1.0, 2.0]
    G = exported["G"]

    sys = CurrentCollector()
    model = SimulationModel(domain, sys, context = DefaultContext())

    # State is dict with pressure in each cell
    phi = 1.
    init = Dict(:Phi => phi)
    state0 = setup_state(model, init)
    state0[:Phi][1] = 2  # Endrer startverdien, skal ikke endre svaret
    
    dirichlet = nothing
    neumann = nothing
    # set up boundary conditions
    T = model.domain.discretizations.charge_flow.conn_data[1].T
    nc = length(domain.grid.volumes)
    boundary_cells = [1, nc]
    boundary_values = [4, -4]
    T_hfs = [1/2, 1/2]
    dirichlet = DirichletBC(boundary_cells, boundary_values, T_hfs)
    neumann = vonNeumannBC([10, nc-9], [1, -1])
    forces = (neumann=neumann, dirichlet= dirichlet,)
    
    # Model parameters
    parameters = setup_parameters(model)

    return (state0, model, parameters, forces, timesteps, G)
end

function get_bccc_struct(name)
    fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, ".mat")
    @debug "Reading MAT file $fn..."
    exported = MAT.matread(fn)
    @debug "File read complete. Unpacking data..."

    bccells = copy((exported["bccells"])')
    bcfaces = copy((exported["bcfaces"])')

    bccells = Int64.(bccells)
    bcfaces = Int64.(bcfaces)
    return (bccells, bcfaces)
end

function get_cc_grid(name, extraout = false)
    fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, ".mat")
    @debug "Reading MAT file $fn..."
    exported = MAT.matread(fn)
    @debug "File read complete. Unpacking data..."

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
    cond = ones(size((exported["rock"]["perm"])')) # Conductivity σ, corresponding to permeability

    volumes = vec(exported["G"]["cells"]["volumes"])

    @debug "Data unpack complete. Starting transmissibility calculations."
    # Deal with face data
    T_hf = compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, cond, N)
    T = compute_face_trans(T_hf, N)

    G = MinimalECTPFAGrid(volumes, N)
    z = nothing
    g = nothing

    ft = ChargeFlow()
    # ??Hva gjør SPU og TPFA??
    flow = TwoPointPotentialFlow(SPU(), TPFA(), ft, G, T, z, g)
    disc = (charge_flow = flow,)
    D = DiscretizedDomain(G, disc)

    if extraout
        return (D, exported)
    else
        return D
    end
end
