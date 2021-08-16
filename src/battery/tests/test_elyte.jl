#=
Electectrolyte
A electrolyte with coupleded potentials (φ, C, T) and |k∇φ|^2-terms
=#
using Terv

ENV["JULIA_DEBUG"] = Terv;

##

function plot_elyte()
    name="square_current_collector"
    domain = get_cc_grid(name=name)
    sys = TestElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())
    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryC] = BoundaryPotential{C}()
    S[:BoundaryT] = BoundaryPotential{T}()

    return plot_graph(model)
end

plot_elyte()

##

function test_elyte()
    name="square_current_collector_10by10"
    bcells, T_hf = get_boundary(name)
    one = ones(size(bcells))
    bcells = [bcells..., (bcells .+ 9)...]
    T_hf = [T_hf..., T_hf...]
    domain, exported = 
        get_cc_grid(name=name, extraout=true, bc=bcells, b_T_hf=T_hf, tensor_map=true)
    t = LinRange(0, 10, 20)
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
        :T                      => 298.,
        :ThermalConductivity    => 6e-05,
        :BoundaryPhi            => [one..., 0*one...],
        :BoundaryC              => [10*one..., one...],
        :BoundaryT              => [273 .* one..., 300 .* one...]
    )

    state0 = setup_state(model, init)

    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    cfg[:info_level] = 2
    cfg[:debug_level] = 2
    states, report = simulate(sim, timesteps, config = cfg)

    return G, states, model, sim
end

G, states, model, sim = test_elyte();
##

f = plot_interactive(G, states)
display(f)
##

accstates = []
for i in 1:18
    state = Dict{Symbol, Any}()
    state[:MassAcc] = (states[i+1][:Mass] .- states[i][:Mass]) # /dt
    push!(accstates, state)
end

f = plot_interactive(G, accstates)
display(f)
##
