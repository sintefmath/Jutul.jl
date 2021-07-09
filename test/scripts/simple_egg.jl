using Terv
using Statistics, DataStructures, LinearAlgebra, Krylov
ENV["JULIA_DEBUG"] = Terv
# ENV["JULIA_DEBUG"] = nothing
##
# casename = "simple_egg"
casename = "egg"
# casename = "egg_ministeps"

# casename = "gravity_test"
# casename = "bl_wells"
# casename = "bl_wells_mini"

# casename = "single_inj_single_cell"
# casename = "intermediate"
# casename = "mini"
simple_well = false
block_backend = true
use_groups = false
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
    # ϵ = 0.5
    # @. s0 += [ϵ; -ϵ]
    init = Dict(:Pressure => p0, :Saturations => s0)
    # state0r = setup_state(model, init)

    ## Model parameters
    param_res = setup_parameters(model)
    param_res[:reference_densities] = vec(rhoS)
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

if include_wells_as_blocks
    w_context = res_context
else
    w_context = DefaultContext()
end

initializer[:Reservoir] = init
forces[:Reservoir] = nothing
models[:Reservoir] = model

slw = 1.0
slw = 0.0
w0 = Dict(:Pressure => mean(init[:Pressure]), :TotalMassFlux => 1e-12, :Saturations => [slw, 1-slw])

well_symbols = map((x) -> Symbol(x["name"]), vec(mrst_data["W"]))
num_wells = length(well_symbols)

well_parameters = Dict()
controls = Dict()
sys = model.system
for i = 1:num_wells
    sym = well_symbols[i]

    wi, wdata = get_well_from_mrst_data(mrst_data, sys, i, 
            extraout = true, volume = 1e-3, simple = simple_well, context = w_context)

    sv = wi.secondary_variables
    sv[:PhaseMassDensities] = model.secondary_variables[:PhaseMassDensities]
    sv[:PhaseViscosities] = model.secondary_variables[:PhaseViscosities]

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
    param_w[:tolerances][:mass_conservation] = 1e-1
    param_w[:reference_densities] = vec(param_res[:reference_densities])

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

mmodel = MultiModel(convert_to_immutable_storage(models), groups = groups, context = outer_context)
# Set up joint state and simulate
state0 = setup_state(mmodel, initializer)
forces[:Facility] = facility_forces

parameters = setup_parameters(mmodel)
parameters[:Reservoir] = param_res

for w in well_symbols
    parameters[w] = well_parameters[w]
end  

sim = Simulator(mmodel, state0 = state0, parameters = deepcopy(parameters))
# dt = [1.0]
# dt = [1.0, 1.0, 10.0, 10.0, 100.0]*3600*24#
dt = timesteps
# dt = dt[1:20]
# dt = dt[1:3]
# dt[1] = 1
# 

atol = 1e-3
rtol = 1e-3
max_it = 100
if use_groups
    well_precond = TrivialPreconditioner()
    res_precond = TrivialPreconditioner()

    res_precond = ILUZeroPreconditioner()
    well_precond = LUPreconditioner()
    # well_precond = ILUZeroPreconditioner()
    if !block_backend
        # res_precond = LUPreconditioner()
    end

    group_p = GroupWisePreconditioner([res_precond, well_precond])

    prec = group_p
    # prec = nothing
else
    prec = nothing
end
lsolve = GenericKrylov(dqgmres, verbose = 10, preconditioner = prec, relative_tolerance = rtol, absolute_tolerance = atol, max_iterations = max_it)
lsolve = nothing

states = simulate(sim, dt, forces = forces, info_level = 1, linear_solver = lsolve, max_iterations = max_it)
error("Early termination")

# return (states, mmodel, well_symbols)
# end
##
sim2 = Simulator(model, state0 = state0r)
lsr = GenericKrylov(verbose = 10, preconditioner = res_precond, max_iterations = 100, 
                relative_tolerance = 1e-3, absolute_tolerance = 1e-2)
# lsr = GenericKrylov(verbose = 10, preconditioner = ILUZeroPreconditioner(), max_iterations = 100)

# lsr = GenericKrylov(verbose = 10, max_iterations = 100000)
# lsr = nothing
# lsr = GenericKrylov(verbose = 10, preconditioner = LUPreconditioner(), max_iterations = 100000)

simulate(sim2, dt, linear_solver = lsr)
##
# states, model, well_symbols = run_immiscible_mrst(casename, false)
# nothing
##
d = map((x) -> x[:Reservoir][:Pressure][1], states)
# d = map((x) -> x[:W1][:Saturations][1], states)
d = map((x) -> x[:Reservoir][:Saturations][1], states)
# d = map((x) -> x[:Reservoir][:Saturations][1, end], states)
d = states[end][:Reservoir].Saturations'
# d = states[end][:Reservoir].Pressure

# d = map((x) -> x[:W1][:Pressure][2] - x[:Reservoir][:Pressure][1], states)

using Plots
Plots.plot(d)
##
# Plots.plot()
##

function get_qws(ix)
    s = well_symbols[ix]
    map((x) -> x[:Facility][:TotalSurfaceMassRate][ix]*x[s][:Saturations][1, 1], states)
end

function get_qos(ix)
    s = well_symbols[ix]
    map((x) -> x[:Facility][:TotalSurfaceMassRate][ix]*x[s][:Saturations][2, 1], states)
end

ix = 9:12
w = well_symbols[ix]
# d = map((x) -> x[w][:Saturations][1], states)

s = states
# s = states_analytical
# s = states_swof_bc
d = map((x) -> -x[:Facility][:TotalSurfaceMassRate][ix], s)
d = hcat(d...)'
h = Plots.plot(d)
# ylims!(h, (0, 6e-3))
##
d = map(get_qws, ix)
d = hcat(d...)
h = Plots.plot(-d)
# ylims!(h, (0, 6e-3))
##
d = map(get_qos, ix)
d = hcat(d...)
h = Plots.plot(-d)
# ylims!(h, (0, 6e-3))

##

function get_bhp(ix)
    s = well_symbols[ix]
    map((x) -> x[s][:Pressure][1], states)
end

ix = 1:8
d = map(get_bhp, ix)
d = hcat(d...)
h = Plots.plot(d)
# ylims!(h, (0, 6e-3))
##
# svar = model.models.Reservoir.secondary_variables
# kr = svar[:RelativePermeabilities]


krw = kr.interpolators[1]
kro = kr.interpolators[2]

sw = collect(-0.2:0.1:1.2)
so = 1 .- sw
Plots.plot(sw, krw.(sw))
Plots.plot!(sw, kro.(so))

##
using ForwardDiff
krw = kr.interpolators[1]
kro = kr.interpolators[2]


##
so = ForwardDiff.Dual(0.80, -1)
kro(so)
##
sw = ForwardDiff.Dual(0.2, 1)
krw(sw)
##
sw = collect(0:0.01:1)
so = 1 .- sw
der = (x) -> x.partials[1]
##
fo = (s) -> kro(ForwardDiff.Dual(s, -1))
ko = fo.(so)
Plots.plot(so, value.(ko), title = "kro")
Plots.plot!(so, der.(ko))
##
fw = (s) -> krw(ForwardDiff.Dual(s, 1))
kw = fw.(sw)
dw = der.(kw)
Plots.plot(sw, value.(kw), title = "krw")
Plots.plot!(sw, dw)