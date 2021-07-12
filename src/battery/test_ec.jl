using Terv
using Test

ENV["JULIA_DEBUG"] = Terv;


function test_ec()
    name="square_current_collector"
    domain, exported = get_cc_grid(MixedFlow(), name=name, extraout=true)
    timesteps = LinRange(0, 10, 10)[2:end]
    G = exported["G"]
    
    sys = ECComponent()
    model = SimulationModel(domain, sys, context = DefaultContext())

    # State is dict with pressure in each cell
    phi = 0.
    c = 0.
    init = Dict(:Phi => phi, :C => c)
    state0 = setup_state(model, init)
    
    # parameters = setup_parameters(model)

    # # set up boundary conditions

    # # Set 1 of boudary conditions
    # bc_phi = DirichletBC{Phi}([1], [1], [2])
    # bc_c = DirichletBC{C}([1], [1], [2])
    # forces = (bc_phi=bc_phi, bc_c=bc_c )

    # # Set 2 of boudary conditions

    bcells, T = get_boundary(name)
    one = ones(size(bcells))

    dirichlet_phi = DirichletBC{Phi}(bcells, 0*one, T)
    neumann_phi = vonNeumannBC{ChargeAcc}(bcells.+9, one)
    dirichlet_c = DirichletBC{C}(bcells, one, T)
    neumann_c = vonNeumannBC{MassAcc}(bcells.+9, one)

    forces = (
        neumann_phi     = neumann_phi, 
        dirichlet_phi   = dirichlet_phi, 
        neumann_c       = neumann_c,
        dirichlet_c     = dirichlet_c
        )

    parameters = setup_parameters(model)
    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, forces = forces, config = cfg)
    return states, G
end

states, G = test_ec();
##
f = plot_interactive(G, states)

display(f)
print(states[1][:MassAcc])
