using Jutul
using Statistics, DataStructures, LinearAlgebra, Krylov, IterativeSolvers
ENV["JULIA_DEBUG"] = Jutul
ENV["JULIA_DEBUG"] = nothing
##
# casename = "simple_egg"
casename = "egg"
# casename = "egg_with_ws"
models, parameters, initializer, timesteps, forces, mrst_data = setup_case_from_mrst(casename);
push!(models[:Reservoir].output_variables, :Saturations)
sim, cfg = setup_reservoir_simulator(models, initializer, parameters)
dt = timesteps

states, reports = simulate(sim, dt, forces = forces, config = cfg);
error("Early termination")
##
res_states = map((x) -> x[:Reservoir], states)
res_states = [Dict(:permx => mrst_data["rock"]["perm"][:, 1])]
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
##
function get_qws(ix)
    s = well_symbols[ix]
    map((x) -> x[:Facility][:TotalSurfaceMassRate][ix]*x[s][:Saturations][1, 1], states)
end

function get_qos(ix)
    s = well_symbols[ix]
    map((x) -> x[:Facility][:TotalSurfaceMassRate][ix]*x[s][:Saturations][2, 1], states)
end

function get_bhp(ix)
    s = well_symbols[ix]
    map((x) -> x[s][:Pressure][1], states)
end

d = map((x) -> x[:Reservoir][:Pressure][1], states)
# d = map((x) -> x[:W1][:Saturations][1], states)
# d = map((x) -> x[:Reservoir][:Saturations][1], states)
# d = map((x) -> x[:Reservoir][:Saturations][1, end], states)
# d = states[end][:Reservoir].Saturations'
# d = states[end][:Reservoir].Pressure

# d = map((x) -> x[:W1][:Pressure][2] - x[:Reservoir][:Pressure][1], states)
wd = mrst_data["well_data"]
injectors = findall(map((w) -> w["sign"] > 0, mrst_data["W"]))
producers = findall(map((w) -> w["sign"] <= 0, mrst_data["W"]))

t = cumsum(dt)./(3600*24);
simulators = ["Julia", "MRST"]
using GLMakie, ColorSchemes
## Plot injector bhp
colors = ColorSchemes.Paired_8.colors
fig = Figure(resolution = (1000, 500))
ax = Axis(fig[1:2, 1], ylabel = "Bottom-hole pressure (bar)", xlabel = "Time (days)")
bhp_m = (wd["bhp"][:, injectors])
bhp = hcat(map(get_bhp, injectors)...)

for (i, wix) in enumerate(injectors)
    GLMakie.lines!(ax, t, bhp[:, i]/1e5, width = 2, color = colors[i])
    GLMakie.scatter!(ax, t, bhp_m[:, i]/1e5, color = colors[i], markersize = 5)
end

labels = String.(well_symbols[injectors])
elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
title = "Injector"
Legend(fig[1,2], elements, labels, title)

elements = [MarkerElement(color = :black, marker = Circle, markersize = 5),
            LineElement(strokewidth  = 5)]
title = "Simulator"
Legend(fig[2,2], elements, simulators, title)
display(fig)
## Plot water rates
colors = ColorSchemes.Paired_4.colors

rhos = mrst_data["fluid"]["rhoS"]
fig = Figure(resolution = (1000, 500))
ax = Axis(fig[1:2, 1], ylabel = "Water production (m³/s)", xlabel = "Time (days)")
qw_m = -(wd["qWs"][:, producers])
qw = -hcat(map(get_qws, producers)...)./rhos[1]

for (i, wix) in enumerate(producers)
    GLMakie.lines!(ax, t, qw_m[:, i], width = 2, color = colors[i])
    GLMakie.scatter!(ax, t, qw[:, i], color = colors[i], markersize = 5)
end

labels = String.(well_symbols[producers])
elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
title = "Producer"
Legend(fig[1,2], elements, labels, title)

elements = [MarkerElement(color = :black, marker = Circle, markersize = 5),
            LineElement(strokewidth  = 5)]
title = "Simulator"
Legend(fig[2,2], elements, simulators, title)
display(fig)
## Plot Oil rates
fig = Figure(resolution = (1000, 500))
ax = Axis(fig[1:2, 1], ylabel = "Oil production (m³/s)", xlabel = "Time (days)")
qo_m = -(wd["qOs"][:, producers])
qo = -hcat(map(get_qos, producers)...)./rhos[2]

for (i, wix) in enumerate(producers)
    GLMakie.lines!(ax, qo_m[:, i], width = 2, color = colors[i])
    GLMakie.scatter!(ax, qo[:, i], color = colors[i], markersize = 5)
end

labels = String.(well_symbols[producers])
elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
title = "Producer"
Legend(fig[1,2], elements, labels, title)

elements = [MarkerElement(color = :black, marker = Circle, markersize = 5),
            LineElement(strokewidth  = 5)]
title = "Simulator"
Legend(fig[2,2], elements, simulators, title)
display(fig)
