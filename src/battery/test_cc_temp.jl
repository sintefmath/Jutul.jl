#=
Current collector with temperature
A conductor with temperature, where the change in temperature is given
by the current density |j|^2
=#
using Terv

ENV["JULIA_DEBUG"] = Terv;


function test_cc(name="square_current_collector")
    domain, exported = get_cc_grid(name=name, extraout=true, bc=[1, 100], b_T_hf=[2., 2.])
    t = LinRange(0, 0.1, 20)
    timesteps = diff(t)
    G = exported["G"]

    sys = CurrentCollectorT()
    model = SimulationModel(domain, sys, context = DefaultContext())

    # State is dict with pressure in each cell
    phi = 0.
    boudary_phi = [1., 2.]
    boudary_T = [1., 1.]

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()

    init = Dict(:Phi => phi, :BoundaryPhi=>boudary_phi, :BoundaryT=>boundary_T)
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
