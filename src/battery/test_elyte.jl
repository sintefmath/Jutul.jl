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

    S = model.secondary_variables
    bcells = [1, 100]
    Thf = [2, 2]
    S[:BoundaryPhi] = BoundaryPotential{Phi}(bcells, Thf)
    S[:BoundaryC] = BoundaryPotential{Phi}(bcells, Thf)
    S[:BoundaryT] = BoundaryPotential{T}(bcells, Thf)

    # S[:BCCharge] = BoundaryCurrent{ChargeAcc}([])
    # S[:BCCMass] = BoundaryCurrent{ChargeAcc}([])
    # S[:BCCEnergy] = BoundaryCurrent{ChargeAcc}([])

    init = Dict(
        :Phi                    => 1.,
        :C                      => 1.,
        :T                      => 1.,
        :Conductivity           => 1.,
        :Diffusivity            => 1.,
        :ThermalConductivity    => 1., 
        :ConsCoeff              => 1.,
        :BoundaryPhi            => [1., 2.],
        :BoundaryC              => [1., 2.],
        :BoundaryT              => [1., 2.], 

        # :BCCharge               => [],
        # :BCCMass                => [],
        # :BCEnergy               => [],
        )

    state0 = setup_state(model, init)

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
