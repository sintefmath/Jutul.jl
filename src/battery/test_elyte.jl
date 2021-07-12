using Terv
ENV["JULIA_DEBUG"] = Terv;

##

function plot_elyte()
    name="square_current_collector"
    domain = get_cc_grid(MixedFlow(), name=name)
    sys = TestElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())
    plot_graph(model)
end

plot_elyte()

##

function test_elyte()
    name="square_current_collector"
    domain, exported = get_cc_grid(MixedFlow(), name=name, extraout=true)
    timesteps = LinRange(0, 1, 10)[2:end]
    G = exported["G"]
    sys = TestElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())
    phi0 = 0.
    c0 = 1.
    T0 = 1.
    κ = 1.
    init = Dict(:Phi=>phi0, :C=>c0, :T=>T0, :Conductivity=>κ, :ConsCoeff=>1)
    state0 = setup_state(model, init)
    parameters = setup_parameters(model)

    bc_phi = DirichletBC{Phi}([10], [phi0], [2])
    bc_c = DirichletBC{C}([10], [c0], [2])
    forces = (bc_phi=bc_phi, bc_c=bc_c)

    bc_c2 = vonNeumannBC{MassAcc}([1, 100], [1, -1])

    forces = (forces..., bc_c2=bc_c2)

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, forces=forces, config = cfg)

    return G, states, model, sim
end

G, states, model, sim = test_elyte();
##

f = plot_interactive(G, states)
display(f)
