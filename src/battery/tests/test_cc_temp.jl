#=
Current collector with temperature
A conductor with temperature, where the change in temperature is given
by the current density |j|^2
=#
using Terv

ENV["JULIA_DEBUG"] = Terv;

##

function test_ccT(name="square_current_collector")
    bc=[1, 9]
    b_T_hf=[2., 2.]
    domain, exported = get_cc_grid(
        name=name, extraout=true, bc=bc, b_T_hf=b_T_hf, tensor_map=true
        )
    time = 1e2
    t = LinRange(0, time, 10)
    timesteps = diff(t)
    G = exported["G"]

    sys = CurrentCollectorT()
    model = SimulationModel(domain, sys, context = DefaultContext())

    # State is dict with pressure in each cell
    phi = 1.
    T0 = 1.
    boundary_phi = [1., 2.]
    boundary_T = [1., 1.]
    κ = 1
    λ = 1

    init = Dict(
        :Phi                    => phi,
        :T                      => T0,                     
        :Conductivity           => κ,
        :ThermalConductivity    => λ,
        :BoundaryPhi            => boundary_phi, 
        :BoundaryT              => boundary_T
        )
    
    state0 = setup_state(model, init)
        
    # Model parameters
    parameters = setup_parameters(model)

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    cfg[:info_level] = 2
    states, report = simulate(sim, timesteps, config = cfg)
    return state0, states, model, G
end

state0, states, model, G = test_ccT();

##

f = plot_interactive(G, states)
display(f)
