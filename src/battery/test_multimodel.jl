#=
Electro-Chemical component
A component with electric potential, concentration and temperature
The different potentials are independent (diagonal onsager matrix),
and conductivity, diffusivity is constant.
=#
using Terv
using MAT
ENV["JULIA_DEBUG"] = Terv;


function test_ac()
    name="model1d"
    fn = string(dirname(pathof(Terv)), "/../data/testgrids/", name, ".mat")
    exported_all = MAT.matread(fn)
    exported = exported_all["model1d"]["NegativeElectrode"]["CurrentCollector"];
    ## for boundary
    bcfaces = [1]
    # bcfaces = Int64.(bfaces)
    T_all = exported["operators"]["T_all"]
    N_all = Int64.(exported["G"]["faces"]["neighbors"])
    isboundary = (N_all[bcfaces,1].==0) .| (N_all[bcfaces,2].==0)
    @assert all(isboundary)
    bccells = N_all[bcfaces,1] + N_all[bcfaces,2]
    T_hf   = T_all[bcfaces]
    bcvalue = ones(size(bccells))


    domain = exported_model_to_domain(exported)
    timesteps = LinRange(0, 10, 10)[2:end]
    #G = exported["G"]
    
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

    S[:BCCharge] = BoundaryCurrent{ChargeAcc}(bccells.+9)
    S[:BCMass] = BoundaryCurrent{MassAcc}(bccells.+9)
    S[:BCEnergy] = BoundaryCurrent{EnergyAcc}(bccells.+9)

    phi0 = 1.
    init = Dict(
        :Phi                    => phi0,
        :C                      => C0,
        :T                      => T0,
        :Conductivity           => σ,
        :Diffusivity            => D,
        :ThermalConductivity    => λ,
        :BoundaryPhi            => bcvalue, 
        :BoundaryC              => bcvalue, 
        :BoundaryT              => bcvalue,
        :BCCharge               => bcvalue,
        :BCMass                 => bcvalue,
        :BCEnergy               => bcvalue,
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