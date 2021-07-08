using Terv

ENV["JULIA_DEBUG"] = Terv;

function test_elyte()
    name="square_current_collector"
    domain, exported = get_cc_grid(MixedFlow(), name=name, extraout=true)
    timesteps = 1:10
    G = exported["G"]
    sys = TestElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())

    phi0 = 0.
    c0 = 0.
    init = Dict(:Phi => phi0, :C => c0)
    state0 = setup_state(model, init)
    parameters = setup_parameters(model)
    forces = (bc=nothing,) # Defaul no flux bc's

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, forces=forces, config = cfg)

    return states
end

test_elyte()

