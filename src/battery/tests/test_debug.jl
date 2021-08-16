#=
A script to lay out the best way to debug
=#

# Start by compiling the needed code, with breakpoints
using Terv

##

function test_cc()
    name="square_current_collector_10by10"
    domain, exported = get_cc_grid(
        name=name, extraout=true, bc=[1, 99], b_T_hf=[2., 2.], tensor_map=false
        )
    timesteps = diff(LinRange(0, 10, 100))
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
    println("done")
end

##

# Use the @run macro instaed of "run debugger"
@run test_cc()
