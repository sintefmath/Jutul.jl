#=
Simple current collector
A conductro with constant conductivity
=#
using Terv

ENV["JULIA_DEBUG"] = Terv;


function test_cc(name="square_current_collector")
    domain, exported = get_cc_grid(name=name, extraout=true, bc=[1, 9], b_T_hf=[2., 2.])
    timesteps = [10.,]
    G = exported["G"]

    sys = CurrentCollector()
    model = SimulationModel(domain, sys, context = DefaultContext())

    # State is dict with pressure in each cell
    phi = 1.
    boudary_phi = [1., 2.]

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()

    init = Dict(:Phi => phi, :BoundaryPhi=>boudary_phi, :Conductivity=>1.)
    state0 = setup_state(model, init)
        
    # Model parameters
    parameters = setup_parameters(model)
    parameters[:tolerances][:default] = 1e-8

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states, _ = simulate(sim, timesteps, config = cfg)
    return state0, states, model, G
end

state0, states, model, G = test_cc();
##

# Can't plot if the first value in state does not match the grid
s = [Dict(k => state[k] for k in keys(state) if k != :TPkGrad_Phi) for state in states]
f = plot_interactive(G, s)
display(f)

##

function test_mixed_bc()
    name="square_current_collector"
    bcells, T_hf = get_boundary(name)
    domain, exported = get_cc_grid(extraout=true, name=name, bc=bcells, b_T_hf=T_hf)
    G = exported["G"]
    timesteps = diff(1:5)

    sys = CurrentCollector()
    model = SimulationModel(domain, sys, context = DefaultContext())

    
    # set up boundary conditions
    one = ones(size(bcells))

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BCCharge] = BoundaryCurrent{ChargeAcc}(2 .+bcells)

    phi0 = 1.
    init = Dict(
        :Phi            => phi0, 
        :BoundaryPhi    => one, 
        :BCCharge       => one,
        :Conductivity   => one
        )
    state0 = setup_state(model, init)
    parameters = setup_parameters(model)

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    cfg[:debug_level] = 2
    cfg[:info_level] = 2
    states = simulate(sim, timesteps, config = cfg)

    return states, G
end


states, G = test_mixed_bc();
##

s = [Dict(k => state[k] for k in keys(state) if k != :TPkGrad_Phi) for state in states]
f = plot_interactive(G, s)
display(f)
