#=
Electro-Chemical component
A component with electric potential, concentration and temperature
The different potentials are independent (diagonal onsager matrix),
and conductivity, diffusivity is constant.
=#
using Terv

ENV["JULIA_DEBUG"] = Terv;


function test_ac()
    name="square_current_collector"
    bcells, T_hf = get_boundary(name)
    one = ones(size(bcells))
    domain, exported = get_cc_grid(MixedFlow(), name=name, extraout=true, bc=bcells, b_T_hf=T_hf)
    timesteps = LinRange(0, 10, 10)[2:end]
    G = exported["G"]
    
    # sys = ECComponent()
    # sys = ACMaterial();
    sys = Grafite()
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

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryC] = BoundaryPotential{Phi}()
    S[:BoundaryT] = BoundaryPotential{T}()

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

states, G = test_ac();
##
f = plot_interactive(G, states);
display(f)