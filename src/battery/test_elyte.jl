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

    return plot_graph(model)
end

plot_elyte()

##

P, S = get_tensorprod()

##

function test_elyte()
    name="square_current_collector"
    bcells, T_hf = get_boundary(name)
    one = ones(size(bcells))
    bcells = [bcells..., (bcells .+ 9)...]
    T_hf = [T_hf..., T_hf...]
    domain, exported = get_cc_grid(
        MixedFlow(), name=name, extraout=true, bc=bcells, b_T_hf=T_hf
        )
    t = LinRange(0, 1, 200)
    timesteps = diff(t)
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
        :BoundaryPhi            => [one..., 0*one...],
        :BoundaryC              => [one..., 0*one...],
        :BoundaryT              => [273 .* one..., 300 .* one...]
        # :BCCharge               => one,
        # :BCMass                 => one,
        # :BCEnergy               => one,
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
