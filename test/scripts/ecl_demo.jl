using Jutul
using Statistics, DataStructures, LinearAlgebra, Krylov, IterativeSolvers
# ENV["JULIA_DEBUG"] = Jutul
ENV["JULIA_DEBUG"] = nothing

casename = "sleipner_ecl_bo"
# casename = "olympus_simple"
casename = "spe1"
models, parameters, initializer, timesteps, forces = setup_case_from_mrst(casename, block_backend = true);
##
sim, cfg = setup_reservoir_simulator(models, initializer, parameters)

states, reports = simulate(sim, dt, forces = forces, config = cfg);
error()
## Plotting
res_states = map((x) -> x[:Reservoir], states)
g = MRSTWrapMesh(mrst_data["G"])

fig, ax = plot_interactive(g, res_states, colormap = :roma)
w_raw = mrst_data["W"]
for w in w_raw
    if w["sign"] > 0
        c = :midnightblue
    else
        c = :firebrick
    end
    plot_well!(ax, g, w, color = c, textscale = 0*5e-2)
end
display(fig)