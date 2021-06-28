using Terv
using Statistics, DataStructures
ENV["JULIA_DEBUG"] = Terv
# ENV["JULIA_DEBUG"] = nothing
##
# casename = "simple_egg"
casename = "egg"
# casename = "gravity_test"
# casename = "bl_wells"
# casename = "bl_wells_mini"

# casename = "single_inj_single_cell"
# casename = "intermediate"
# casename = "mini"
simple_well = false


G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true, fuse_flux = false)
## Set up initializers
models = OrderedDict()
initializer = Dict()
forces = Dict()
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
model = SimulationModel(G, sys)

rho = ConstantCompressibilityDensities(sys, p, rhoS, c)

swof = f["swof"]

if isempty(swof)
    kr = BrooksCoreyRelPerm(sys, nkr)
else
    s = swof[:, 1]
    krt = vcat(swof[:, 2]', 1 .- swof[:, 3]')
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
init = Dict(:Pressure => p0, :Saturations => s0)
state0r = setup_state(model, init)

## Model parameters
param_res = setup_parameters(model)
param_res[:reference_densities] = vec(rhoS)

dt = mrst_data["dt"]
if isa(dt, Real)
    dt = [dt]
end
timesteps = vec(dt)
sim = Simulator(model, state0 = state0r, parameters = param_res)

# simulate(sim, timesteps)

initializer[:Reservoir] = init
forces[:Reservoir] = nothing
models[:Reservoir] = model

## Set up injector

# Initial condition for all wells
slw = 1.0
slw = 0.0
w0 = Dict(:Pressure => mean(p0), :TotalMassFlux => 1e-12, :Saturations => [slw, 1-slw])

well_symbols = map((x) -> Symbol(x["name"]), vec(mrst_data["W"]))
num_wells = length(well_symbols)

well_parameters = Dict()
controls = Dict()
for i = 1:num_wells
    sym = well_symbols[i]

    wi, wdata = get_well_from_mrst_data(mrst_data, sys, i, extraout = true, volume = 1e-3, simple = simple_well)

    sv = wi.secondary_variables
    sv[:PhaseMassDensities] = rho
    sv[:PhaseViscosities] = mu

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
        ctrl = InjectorControl(target, wdata["compi"])
    else
        println("Producer")
        ctrl = ProducerControl(target)
    end
    param_w = setup_parameters(wi)
    param_w[:tolerances][:mass_conservation] = 1e-1
    param_w[:reference_densities] = vec(rhoS)

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
fstate = setup_state(WG, F0)

facility_forces = build_forces(WG, control = controls)
models[:Facility] = WG
initializer[:Facility] = F0
##

mmodel = MultiModel(convert_to_immutable_storage(models))
# mmodel = MultiModel((Reservoir = model, Injector = Wi, Producer = Wp, Facility = WG))
# Set up joint state and simulate
state0 = setup_state(mmodel, initializer)
forces[:Facility] = facility_forces
# forces = Dict(:Reservoir => nothing, :Facility => facility_forces, :Injector => nothing, :Producer => nothing)

parameters = setup_parameters(mmodel)
parameters[:Reservoir] = param_res

for w in well_symbols
    parameters[w] = well_parameters[w]
end

sim = Simulator(mmodel, state0 = state0, parameters = deepcopy(parameters))
# dt = [1.0]
# dt = [1.0, 1.0, 10.0, 10.0, 100.0]*3600*24
dt = timesteps
states = simulate(sim, dt, forces = forces, info_level = 1)
nothing
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
d = map((x) -> -x[:Facility][:TotalSurfaceMassRate][ix], states)
d = hcat(d...)'
h = Plots.plot(d)
# ylims!(h, (0, 6e-3))
##
d = map(get_qws, ix)
d = hcat(d...)
h = Plots.plot(-d)
ylims!(h, (0, 6e-3))
##
d = map(get_qos, ix)
d = hcat(d...)
h = Plots.plot(-d)
ylims!(h, (0, 6e-3))

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