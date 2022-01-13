using Jutul
using Statistics, DataStructures, LinearAlgebra, Krylov, IterativeSolvers
ENV["JULIA_DEBUG"] = Jutul
ENV["JULIA_DEBUG"] = nothing
##
# casename = "simple_egg"
casename = "egg"
casename = "egg_with_ws"
# casename = "egg_ministeps"

# casename = "gravity_test"
# casename = "bl_wells"
# casename = "bl_wells_mini"

# casename = "single_inj_single_cell"
# casename = "intermediate"
# casename = "mini"
# casename = "spe10_2ph_1"
# casename = "spe10_2ph_1_85"
# casename = "olympus_simple"

simple_well = false
block_backend = true
# block_backend = false
use_groups = true
use_schur = true

# Need groups to work.
use_schur = use_schur && use_groups

include_wells_as_blocks = use_groups && simple_well && false    
G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true, fuse_flux = false)
function setup_res(G, mrst_data; block_backend = false, use_groups = false)
    ## Set up reservoir part
    f = mrst_data["fluid"]
    p = vec(f["p"])
    c = vec(f["c"])
    mu = vec(f["mu"])
    nkr = vec(f["nkr"])
    rhoS = vec(f["rhoS"])

    water = AqueousPhase()
    oil = LiquidPhase()
    sys = ImmiscibleSystem([water, oil])
    bctx = DefaultContext(matrix_layout = BlockMajorLayout())
    # bctx = DefaultContext(matrix_layout = UnitMajorLayout())
    dctx = DefaultContext()

    if block_backend && use_groups
        res_context = bctx
    else
        res_context = dctx
    end

    model = SimulationModel(G, sys, context = res_context)
    rho = ConstantCompressibilityDensities(sys, p, rhoS, c)

    if haskey(f, "swof")
        swof = f["swof"]
    else
        swof = []
    end

    if isempty(swof) # || true
        kr = BrooksCoreyRelPerm(sys, nkr)
    else
        display(swof)
        s, krt = preprocess_relperm_table(swof)
        kr = TabulatedRelPermSimple(s, krt)
    end
    mu = ConstantVariables(mu)

    p = model.primary_variables
    # p[:Pressure] = Pressure(50*1e5, 0.2)
    s = model.secondary_variables
    s[:PhaseMassDensities] = rho
    s[:RelativePermeabilities] = kr
    s[:PhaseViscosities] = mu

    ##
    state0 = mrst_data["state0"]
    p0 = state0["pressure"]
    if isa(p0, AbstractArray)
        p0 = vec(p0)
    else
        p0 = [p0]
    end
    s0 = state0["s"]'
    if false
        @warn "ϵ is on"
        ϵ = 0.5
        @. s0 += [ϵ; -ϵ]
        end
    init = Dict(:Pressure => p0, :Saturations => s0)
    # state0r = setup_state(model, init)

    ## Model parameters
    param_res = setup_parameters(model)
    param_res[:reference_densities] = vec(rhoS)
    param_res[:tolerances][:default] = 0.01
    param_res[:tolerances][:mass_conservation] = 0.01

    return (model, init, param_res)
end

## Set up initializers
models = OrderedDict()
initializer = Dict()
forces = Dict()

# function run_immiscible_mrst(casename, simple_well = false)
model, init, param_res = setup_res(G, mrst_data; block_backend = block_backend, use_groups = use_groups)

dt = mrst_data["dt"]
if isa(dt, Real)
    dt = [dt]
end
timesteps = vec(dt)

res_context = model.context
if include_wells_as_blocks
    w_context = res_context
else
    w_context = DefaultContext()
end
# w_context = res_context

initializer[:Reservoir] = init
forces[:Reservoir] = nothing
models[:Reservoir] = model

slw = 1.0
# slw = 0.0
# slw = 0.5
w0 = Dict(:Pressure => mean(init[:Pressure]), :TotalMassFlux => 1e-12, :Saturations => [slw, 1-slw])

well_symbols = map((x) -> Symbol(x["name"]), vec(mrst_data["W"]))
num_wells = length(well_symbols)

well_parameters = Dict()
controls = Dict()
sys = model.system
for i = 1:num_wells
    sym = well_symbols[i]

    wi, wdata = get_well_from_mrst_data(mrst_data, sys, i, 
            extraout = true, volume = 1e-2, simple = simple_well, context = w_context)

    sv = wi.secondary_variables
    sv[:PhaseMassDensities] = model.secondary_variables[:PhaseMassDensities]
    sv[:PhaseViscosities] = model.secondary_variables[:PhaseViscosities]

    pw = wi.primary_variables
    # pw[:Pressure] = Pressure(50*1e5, 0.2)

    models[sym] = wi

    println("$sym")
    t_mrst = wdata["val"]
    is_injector = wdata["sign"] > 0
    is_shut = wdata["status"] < 1
    if wdata["type"] == "rate"
        println("Rate")
        target = SinglePhaseRateTarget(t_mrst, AqueousPhase())
    else
        println("BHP")
        target = BottomHolePressureTarget(t_mrst)
    end

    if is_shut
        println("Shut well")
        ctrl = DisabledControl()
    elseif is_injector
        println("Injector")
        ci = wdata["compi"]
        # ci = [0.5, 0.5]
        ctrl = InjectorControl(target, ci)
    else
        println("Producer")
        ctrl = ProducerControl(target)
    end
    param_w = setup_parameters(wi)
    param_w[:reference_densities] = vec(param_res[:reference_densities])
    param_w[:tolerances][:mass_conservation] = 0.01

    well_parameters[sym] = param_w
    controls[sym] = ctrl
    forces[sym] = nothing
    initializer[sym] = w0
end
##
g = WellGroup(well_symbols)
mode = PredictionMode()
WG = SimulationModel(g, mode)
F0 = Dict(:TotalSurfaceMassRate => 0.0)

facility_forces = build_forces(WG, control = controls)
models[:Facility] = WG
initializer[:Facility] = F0
##
if use_groups
    if include_wells_as_blocks
        groups = repeat([1], length(models))
        groups[end] = 2
    else
        groups = repeat([2], length(models))
        groups[1] = 1
    end
else
    groups = nothing
end
outer_context = DefaultContext()
# outer_context = nothing

if use_schur
    red = :schur_apply
else
    red = nothing
end
mmodel = MultiModel(convert_to_immutable_storage(models), groups = groups, context = outer_context, reduction = red)
# Set up joint state and simulate
state0 = setup_state(mmodel, initializer)
forces[:Facility] = facility_forces

parameters = setup_parameters(mmodel)
parameters[:Reservoir] = param_res

for w in well_symbols
    parameters[w] = well_parameters[w]
end  

# dt = [1.0]
# dt = [1.0, 1.0, 10.0, 10.0, 100.0]*3600*24#
dt = timesteps
# dt = dt[1:2]
# dt = dt[1:20]
# dt = dt[1:3]
# dt[1] = 1
# 


# atol = 0.01
# rtol = 0.01
if use_groups
    well_precond = TrivialPreconditioner()
    res_precond = TrivialPreconditioner()

    res_precond = ILUZeroPreconditioner(right = false)
    # res_precond =  DampedJacobiPreconditioner()
    well_precond = LUPreconditioner()
    # well_precond = ILUZeroPreconditioner()
    if !block_backend
        res_precond = LUPreconditioner()
    end

    if use_schur
        group_p = res_precond
    else
        group_p = GroupWisePreconditioner([res_precond, well_precond])
    end
 
    prec = group_p
    # prec = nothing
else
    prec = LUPreconditioner()
end
using AlgebraicMultigrid
p_solve = AMGPreconditioner(smoothed_aggregation)
# p_solve = LUPreconditioner()
cpr_type = :true_impes
# cpr_type = :quasi_impes
# cpr_type = :none
update_interval = :iteration
update_interval = :ministep
update_interval = :once

prec = CPRPreconditioner(p_solve, strategy = cpr_type, 
                    update_interval = update_interval, partial_update = false)

# prec = ILUZeroPreconditioner()
atol = 1e-12
rtol = 1e-12

atol = 1e-12
rtol = 1e-2
# rtol = 0.01
rtol = 0.05
# rtol = 0.001
# rtol = 1e-2

max_it = 50
# atol = 1e-18
# rtol = 1e-18

# max_it = 200#1000
# max_it = 1
krylov = bicgstab
krylov = dqgmres
krylov = IterativeSolvers.gmres!
# krylov = Krylov.dqgmres
# prec = nothing 
lsolve = GenericKrylov(krylov, verbose = 0, preconditioner = prec, 
                        relative_tolerance = rtol, absolute_tolerance = atol,
                        max_iterations = max_it)
if !use_groups
    # lsolve = nothing
end
m = 20
# m = 100
# m = 6
# m = 1

max_cuts = 0
max_cuts = 5
il = 1
dl = 0
# dt = dt[1:3]
# dt = dt[[1]]
# dt = [1.0, 100.0, 10000.0]
# 24.7 s
# 27.6
# lsolve = nothing
sim = Simulator(mmodel, state0 = state0, parameters = deepcopy(parameters))
cfg = simulator_config(sim, info_level = il, debug_level = dl,
                            max_nonlinear_iterations = m,
                            max_timestep_cuts = max_cuts,
                            output_states = true,
                            linear_solver = lsolve)
@time states, reports = simulate(sim, dt, forces = forces, config = cfg)
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
