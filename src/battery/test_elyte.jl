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
    parameters = setup_parameters(model)

    bcells, T = get_boundary(name)
    one = ones(size(bcells))

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}(bcells, T)
    S[:BCCharge] = BoundaryCurrent{ChargeAcc}(bcells.+9)
    S[:BoundaryC] = BoundaryPotential{Phi}(bcells, T)
    S[:BCMass] = BoundaryCurrent{ChargeAcc}(bcells.+9)

    init = Dict(
        :Phi            => 1.,
        :C              => 1.,
        :Conductivity   => 1.,
        :Diffusivity    => 1.,
        :T              => 1., 
        :ConsCoeff      => 1.,
        :BoundaryPhi    => one, 
        :BCCharge       => one,
        :BoundaryC      => one, 
        :BCMass         => one,
        )

    state0 = setup_state(model, init)
    # return state0

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, config = cfg)

    return G, states, model, sim
end

G, states, model, sim = test_elyte();
##

f = plot_interactive(G, states)
display(f)
