using Jutul
using Statistics, DataStructures, LinearAlgebra, Krylov, IterativeSolvers
ENV["JULIA_DEBUG"] = Jutul
ENV["JULIA_DEBUG"] = nothing

casename = "sleipner_ecl_bo"
casename = "olympus_1_reduced"
# casename = "spe1"
# casename = "C:/Users/olavm/AppData/Local/Temp/qfs_wo.mat"
casename = "C:/Users/olavm/AppData/Local/Temp/buckley_leverett_wo.mat"
casename = "C:/Users/olavm/AppData/Local/Temp/spe10_layer.mat"
casename = "C:/Users/olavm/AppData/Local/Temp/qfs_wo.mat"
# casename = "spe1"
# casename = "spe1_big"
# casename = "spe9"
models, parameters, initializer, dt, forces, mrst_data = setup_case_from_mrst(casename, 
                                block_backend = true, simple_well = false);


sim, cfg = setup_reservoir_simulator(models, initializer, parameters, info_level = 3, 
            max_timestep_cuts = 0, extra_timing = false, method = :cpr);
##
# update_state_dependents!(sim.storage, sim.model, 1.0, forces)
##
states, reports = simulate(sim, dt, forces = forces, config = cfg);
##
error()
## Plotting
using GLMakie, Jutul
res_states = map((x) -> x[:Reservoir], states)
g = MRSTWrapMesh(mrst_data["G"])

fig, ax = plot_interactive(g, res_states, colormap = :turbo)
w_raw = mrst_data["schedule"]["control"][1]["W"]
for w in w_raw
    if w["sign"] > 0
        c = :midnightblue
    else
        c = :firebrick
    end
    plot_well!(ax, g, w, color = c, textscale = 1*5e-2)
end
display(fig);
##
wd = full_well_outputs(sim.model, parameters, states)
time = report_times(reports)
plot_well_results(wd, time)
##
qg = wd[:PRODUCER][Symbol("Surface gas rate")]
qo = wd[:PRODUCER][Symbol("Surface oil rate")]
lines(time./(365), qg./qo)
##
kr = models[:Reservoir].secondary_variables[:RelativePermeabilities]
s = 0:0.01:1;
f = lines(s, kr.krw.(s), label = "krw")
lines!(s, kr.krg.(s), label = "krg")
lines!(s, kr.krow.(s), label = "krow")
scatter!(s, kr.krog.(s), label = "krog")
Legend(f.figure[1, 2], f.axis)
f
##
rsvar = models[:Reservoir].secondary_variables
if haskey(rsvar, :CapillaryPressure)
    pc = rsvar[:CapillaryPressure]
    s = 0:0.01:1;
    f = lines(s, pc.pc[1].(s), label = "pcow")
    lines!(s, pc.pc[2].(s), label = "pcog")
    Legend(f.figure[1, 2], f.axis)
    f
end
##
mg = map(x-> normalize(x[:PRODUCER][:TotalMasses][:, 1], 1), states)
mg = hcat(mg...)
q = map(x -> x[:Facility][:TotalSurfaceMassRate][2], states)
##
qi = mg'.*q
series(qi')