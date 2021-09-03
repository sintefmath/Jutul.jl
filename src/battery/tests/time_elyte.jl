#=
Get timing of the simple elyte model
=#

@time using Terv

##
using MAT
ENV["JULIA_DEBUG"] = 0

function test_simple_elyte()
    model, exported = get_simple_elyte_model()
    sim = get_simple_elyte_sim(model, exported)
    timesteps = exported["schedule"]["step"]["val"][:, 1]
    G = exported["model"]["G"]
  
    cfg = simulator_config(sim)
    cfg[:linear_solver] = nothing
    simulate(sim, timesteps, config = cfg)
end

##
@time test_simple_elyte();

##
@time test_simple_elyte()