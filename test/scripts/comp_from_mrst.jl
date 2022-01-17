using Jutul
using Statistics, DataStructures, LinearAlgebra, Krylov, IterativeSolvers, MultiComponentFlash
ENV["JULIA_DEBUG"] = nothing

casename = "1d_validation"
casename = "1d_validation_water"
# casename = "1d_validation_water_mini"

# casename = "fractures_compositional"
function bench(casename)
    block = true
    # block = false
    sw = true
    sw = false
    mb = 250
    models, parameters, initializer, timesteps, forces, mrst_data = setup_case_from_mrst(casename, block_backend = block, minbatch = mb);
    push!(models[:Reservoir].output_variables, :Saturations)
    sim, cfg = setup_reservoir_simulator(models, initializer, parameters)
    dt = timesteps

    res = simulate(sim, dt, forces = forces, config = cfg);
    return (res[1], res[2], sim, mrst_data)
end
states, reports, sim, mrst_data = bench(casename);
error("All done.")
##
using GLMakie
k = :OverallMoleFractions
k = :Saturations
getdata = x -> x[:Reservoir][k]
z0 = getdata(states[180])

plotvals = Observable(z0)
f = Figure()
ax = Axis(f[1, 1])
series!(plotvals, color = :Set1)
GLMakie.ylims!(0, 1)
f
##
for i = 1:length(states)
    plotvals[] = getdata(states[i])
    sleep(0.01)
end
##
using GLMakie
res_states = map((x) -> x[:Reservoir], states)
g = MRSTWrapMesh(mrst_data["G"])

fig, ax = plot_interactive(g, res_states, colormap = :turbo)
w_raw = mrst_data["W"]
for w in w_raw
    if w["sign"] > 0
        c = :midnightblue
    else
        c = :firebrick
    end
    # plot_well!(ax, g, w, color = c, textscale = 1*5e-2)
end
display(fig) 
##