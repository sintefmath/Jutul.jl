using Terv

export get_test_setup_battery, get_cc_grid, get_bc, test_mixed_boundary_conditions
export get_test_setup_ec_component

function get_test_setup_battery(name="square_current_collector")
    domain, exported = get_cc_grid(name=name, extraout=true)
    timesteps = [1., 2, 3, 4]
    G = exported["G"]

    sys = CurrentCollector()
    model = SimulationModel(domain, sys, context = DefaultContext())

    # State is dict with pressure in each cell
    phi = 1.
    init = Dict(:Phi => phi)
    state0 = setup_state(model, init)
    state0[:Phi][1] = 2  # Endrer startverdien, skal ikke endre svaret
    
    # set up boundary conditions
    nc = length(domain.grid.volumes)
    
    dirichlet = DirichletBC([1, 1], [1, -1], [2, 2])
    neumann = vonNeumannBC([1, nc], [-1, 1])
    forces = (neumann=neumann, dirichlet= dirichlet,)
    
    # Model parameters
    parameters = setup_parameters(model)

    return (state0, model, parameters, forces, timesteps, G)
end

function test_mixed_boundary_conditions()
    name="square_current_collector"
    domain, exported = get_cc_grid(ChargeFlow(), extraout=true, name=name)
    G = exported["G"]
    timesteps = [1.,]

    sys = CurrentCollector()
    model = SimulationModel(domain, sys, context = DefaultContext())

    # State is dict with pressure in each cell
    phi = 1.
    init = Dict(:Phi => phi)
    state0 = setup_state(model, init)
    
    # set up boundary conditions
    
    bcells, T = get_boundary(name)
    one = ones(size(bcells))

    dirichlet = DirichletBC{Phi}(bcells, one, T)
    neumann = vonNeumannBC{Phi}(bcells.+9, one)
    forces = (neumann=neumann, dirichlet= dirichlet,)
    
    # Model parameters
    parameters = setup_parameters(model)

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, forces = forces, config = cfg)

    # Check if the field value increments by one
    # @assert sum(isapprox.(diff(states[1].Phi[1:10]), 1)) == 9
    return states, G
end


function get_test_setup_ec_component()
    domain, exported = get_cc_grid(MixedFlow(), extraout=true)
    timesteps = 1:10
    G = exported["G"]
    
    sys = ECComponent()
    model = SimulationModel(domain, sys, context = DefaultContext())

    # State is dict with pressure in each cell
    phi = 0.
    c = 0.
    init = Dict(:Phi => phi, :C => c)
    state0 = setup_state(model, init)

    # set up boundary conditions
    bc_phi = DirichletBC{Phi}([1], [1], [2])
    bc_c = DirichletBC{C}([1], [1], [2])
    forces = (bc_phi=bc_phi, bc_c=bc_c )
    parameters = setup_parameters(model)

    return (state0, model, parameters, forces, timesteps, G)
end


function get_boundary(name)
    fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, "_T.mat")
    exported = MAT.matread(fn)

    exported
    bccells = copy((exported["bccells"])')
    T = copy((exported["T"])')

    bccells = Int64.(bccells)

    return (bccells, T)
end



function get_cc_grid(
    flow_type=ChargeFlow(); name="square_current_collector", extraout = false
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
    face_centroids = copy((
        exported["G"]["faces"]["centroids"][internal_faces, :])'
        )
    face_areas = vec(exported["G"]["faces"]["areas"][internal_faces])
    face_normals = exported["G"]["faces"]["normals"][internal_faces, :]./face_areas
    face_normals = copy(face_normals')
    cond = ones(size((exported["rock"]["perm"])')) # Conductivity σ, corresponding to permeability
    volumes = vec(exported["G"]["cells"]["volumes"])

    # Deal with face data
    T_hf = compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, cond, N)
    T = compute_face_trans(T_hf, N)

    G = MinimalECTPFAGrid(volumes, N)
    z = nothing
    g = nothing

    ft = flow_type
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
