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
    parameters = setup_parameters(model)
    parameters[:boundary_currents] = (:BCCharge, :BCMass)

    # State is dict with pressure in each cell
    phi0 = 1.
    C0 = 1.
    D = 1.
    σ = 1

    bcells, T = get_boundary(name)
    one = ones(size(bcells))

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}(bcells, T)
    S[:BCCharge] = BoundaryCurrent{ChargeAcc}(bcells.+9)
    S[:BoundaryC] = BoundaryPotential{Phi}(bcells, T)
    S[:BCMass] = BoundaryCurrent{ChargeAcc}(bcells.+9)

    phi0 = 1.
    init = Dict(
        :Phi            => phi0,
        :C              => C0,
        :Conductivity   => σ,
        :Diffusivity    => D,
        :BoundaryPhi    => one, 
        :BCCharge       => one,
        :BoundaryC      => one, 
        :BCMass         => one,
        )

    state0 = setup_state(model, init)

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, config = cfg)
    return states, G
end

states, G = test_ec();
##
f = plot_interactive(G, states);
display(f)