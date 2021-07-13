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
    T0 = 1.
    D = 1.
    σ = 1.
    λ = 1.

    bcells, Thf = get_boundary(name)
    one = ones(size(bcells))

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}(bcells, Thf)
    S[:BoundaryC] = BoundaryPotential{Phi}(bcells, Thf)
    S[:BoundaryT] = BoundaryPotential{T}(bcells, Thf)

    S[:BCCharge] = BoundaryCurrent{ChargeAcc}(bcells.+9)
    S[:BCMass] = BoundaryCurrent{MassAcc}(bcells.+9)
    S[:BCEnergy] = BoundaryCurrent{EnergyAcc}(bcells.+9)

    phi0 = 1.
    init = Dict(
        :Phi                    => phi0,
        :C                      => C0,
        :T                      => T0,
        :Conductivity           => σ,
        :Diffusivity            => D,
        :ThermalConductivity    => λ,
        :BoundaryPhi            => one, 
        :BoundaryC              => one, 
        :BoundaryT              => one,
        :BCCharge               => one,
        :BCMass                 => one,
        :BCEnergy               => one,
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