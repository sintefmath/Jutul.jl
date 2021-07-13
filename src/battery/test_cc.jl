using Terv
using Test

ENV["JULIA_DEBUG"] = Terv;


function test_cc(name="square_current_collector")
    domain, exported = get_cc_grid(name=name, extraout=true)
    timesteps = [1., ]
    G = exported["G"]

    sys = CurrentCollector()
    model = SimulationModel(domain, sys, context = DefaultContext())

    # State is dict with pressure in each cell
    phi = 0.

    boudary_phi = [0., 1]
    b_cells = [1, 100]
    T_half_face = [2., 2.]
    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}(b_cells, T_half_face)

    init = Dict(:Phi => phi, :BoundaryPhi=>boudary_phi)
    state0 = setup_state(model, init)
        
    # Model parameters
    parameters = setup_parameters(model)

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, config = cfg)
    return state0, states, model, G
end

state0, states, model, G = test_cc();
##
f = plot_interactive(G, states)
display(f)

##

function test_mixed_bc()
    name="square_current_collector"
    domain, exported = get_cc_grid(ChargeFlow(), extraout=true, name=name)
    G = exported["G"]
    timesteps = 1:5

    sys = CurrentCollector()
    model = SimulationModel(domain, sys, context = DefaultContext())

    
    # set up boundary conditions
    bcells, T = get_boundary(name)
    one = ones(size(bcells))

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}(bcells, T)
    S[:BCCharge] = BoundaryCurrent{ChargeAcc}(bcells.+9)

    phi0 = 1.
    init = Dict(
        :Phi            => phi0, 
        :BoundaryPhi    => one, 
        :BCCharge       => one
        )
    state0 = setup_state(model, init)
    parameters = setup_parameters(model)

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, config = cfg)

    return states, G
end


states, G = test_mixed_bc();
##
f = plot_interactive(G, states)
display(f)
