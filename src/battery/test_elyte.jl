using Terv

ENV["JULIA_DEBUG"] = Terv;

function test_elyte()
    name="square_current_collector"
    domain, exported = get_cc_grid(MixedFlow(), name=name, extraout=true)
    timesteps = 1:10
    G = exported["G"]
    sys = TestElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())

    phi0 = 0.
    c0 = 0.
    T0 = 0.
    init = Dict(:Phi => phi0, :C => c0, :T =>T0)
    state0 = setup_state(model, init)
    parameters = setup_parameters(model)

    bc_phi = DirichletBC{Phi}([1], [1], [2])
    bc_c = DirichletBC{C}([1], [1], [2])
    bc_T = DirichletBC{T}([1], [1], [2])
    forces = (bc_phi=bc_phi, bc_c=bc_c, bc_T=bc_T,)

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, forces=forces, config = cfg)

    return G, states
end

G, states = test_elyte();
##

f = plot_interactive(G, states)
display(f)
