#=
Compare the simulation of electrolyte in Julia and Matlab
=#

using Terv
using MAT
ENV["JULIA_DEBUG"] = Terv;

##

function plot_elyte()
    model, _ = get_simple_elyte_model()
    return plot_graph(model)
end

plot_elyte()

##

function test_simple_elyte()
    model, exported = get_simple_elyte_model()
    sim = get_simple_elyte_sim(model, exported)
    timesteps = exported["schedule"]["step"]["val"][:, 1]
    G = exported["model"]["G"]
  
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
    state[:Mass] = (states[i+1][:Mass] .- states[i][:Mass])
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