using Terv

ENV["JULIA_DEBUG"] = Terv;

##

function plot_elyte()
    name="square_current_collector"
    domain = get_cc_grid(MixedFlow(), name=name)
    sys = TestElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())
    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryC] = BoundaryPotential{Phi}()
    S[:BoundaryT] = BoundaryPotential{T}()

    plot_graph(model)
end

plot_elyte()

##

function test_elyte()
    name="square_current_collector"
    bc = [1, 100]
    T_hf = [2., 2.]
    domain, exported = get_cc_grid(
        MixedFlow(), name=name, extraout=true, bc=bc, b_T_hf=T_hf
        )
    timesteps = LinRange(0, 0.1, 100)[2:end]
    G = exported["G"]
    sys = TestElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())
    parameters = setup_parameters(model)

    S = model.secondary_variables

    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryC] = BoundaryPotential{C}()
    S[:BoundaryT] = BoundaryPotential{T}()


    init = Dict(
        :Phi                    => 1.,
        :C                      => 1.,
        :T                      => 273.,
        :Conductivity           => 1.,
        :Diffusivity            => 1.,
        :ThermalConductivity    => 6e-05, 
        :ConsCoeff              => 1.,
        :BoundaryPhi            => [1., 2.],
        :BoundaryC              => [1., 1.],
        :BoundaryT              => [273., 300.], 
    )

    state0 = setup_state(model, init)

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    states = simulate(sim, timesteps, config = cfg)

    return G, states, model, sim
end

G, states, model, sim = test_elyte();
##

f = plot_interactive(G, states)
display(f)
##
