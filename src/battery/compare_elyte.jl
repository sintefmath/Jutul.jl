#=
Compare the simulation of electrolyte in Julia and Matlab
=#

using Terv
using MAT
ENV["JULIA_DEBUG"] = Terv;

function get_simple_elyte_model()
    name="modelElectrolyte"
    fn = string(dirname(pathof(Terv)), "/../data/models/", name, ".mat")
    exported = MAT.matread(fn)
    ex_model = exported["model"]

    boundary = ex_model["bcstruct"]["dirichlet"]

    b_faces = Int64.(boundary["faces"])
    T_all = ex_model["operators"]["T_all"]
    N_all = Int64.(ex_model["G"]["faces"]["neighbors"])
    isboundary = (N_all[b_faces, 1].==0) .| (N_all[b_faces, 2].==0)
    @assert all(isboundary)
    bc_cells = N_all[b_faces, 1] + N_all[b_faces, 2]
    b_T_hf   = T_all[b_faces]

    domain = exported_model_to_domain(ex_model, bc=bc_cells, b_T_hf=b_T_hf)

    sys = SimpleElyte()
    model = SimulationModel(domain, sys, context = DefaultContext())

    S = model.secondary_variables
    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryC] = BoundaryPotential{C}()

    return model, exported
end

##

function plot_elyte()
    model, _ = get_simple_elyte_model()
    return plot_graph(model)
end

plot_elyte()

##

function test_simple_elyte()
    model, exported = get_simple_elyte_model()
    boundary = exported["model"]["bcstruct"]["dirichlet"]
    b_phi = boundary["phi"][:, 1]
    b_c = boundary["conc"][:, 1]
    init_states = exported["state0"]
    init = Dict(
        :Phi            => init_states["phi"][:, 1],
        :C              => init_states["cs"][1][:, 1],
        :T              => init_states["T"][:, 1],
        :BoundaryPhi    => b_phi,
        :BoundaryC      => b_c,
        )

    state0 = setup_state(model, init)

    parameters = setup_parameters(model)
    parameters[:tolerances][:default] = 1e-8
    parameters[:t] = exported["model"]["sp"]["t"][1]
    parameters[:z] = exported["model"]["sp"]["z"][1]
 
    timesteps = exported["schedule"]["step"]["val"][:, 1]

    G = exported["model"]["G"]
  
    sim = Simulator(model, state0=state0, parameters=parameters)
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    cfg[:info_level] = 2
    cfg[:debug_level] = 2
    states, report = simulate(sim, timesteps, config = cfg)
    rs = exported["states"][:, 1]
    return states, G, model, rs
end

states, G, model, rs = test_simple_elyte();

##

f = plot_interactive(G, states)
display(f)
##

accstates = []
for i in 1:19
    state = Dict{Symbol, Any}()
    state[:MassAcc] = (states[i+1][:MassAcc] .- states[i][:MassAcc])
    push!(accstates, state)
end

f = plot_interactive(G, accstates)
display(f)
##

j2m = Dict{Symbol, String}(
    :C                  => "cs",
    :T                  => "T",
    :Phi                => "phi", 
    :Conductivity       => "conductivity",
    :Diffusivity        => "D",
    :TotalCurrent       => "j",
    :ChargeCarrierFlux  => "LiFlux" 
)
ref_states = get_ref_states(j2m, rs);

##
f = plot_interactive(G, ref_states)
display(f)

##

function print_diff(s, sref, n, k)
    Δ = abs.(1 .- s[n][k] ./ sref[n][k])
    println("Δ = $(maximum(Δ))")
end

ks = (:Phi, :C)

for n in 1:20
    println("Step $n")
    for k in ks
        println(k)
        print_diff(states, ref_states, n, k)
    end
end