using Terv
using Statistics, DataStructures, LinearAlgebra, Krylov, IterativeSolvers
# ENV["JULIA_DEBUG"] = Terv
ENV["JULIA_DEBUG"] = nothing

casename = "egg"

simple_well = false
block_backend = true

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

    if isempty(swof)
        kr = BrooksCoreyRelPerm(sys, nkr)
    else
        s, krt = preprocess_relperm_table(swof)
        kr = TabulatedRelPermSimple(s, krt)
    end
    mu = ConstantVariables(mu)

    p = model.primary_variables
    p[:Pressure] = Pressure(max_rel = 0.2)
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

    ## Model parameters
    param_res = setup_parameters(model)
    param_res[:reference_densities] = vec(rhoS)
    param_res[:tolerances][:default] = 0.01
    param_res[:tolerances][:mass_conservation] = 0.01

    return (model, init, param_res)
end

# Set up initializers
models = OrderedDict()
initializer = Dict()
forces = Dict()

model, init, param_res = setup_res(G, mrst_data; block_backend = block_backend, use_groups = true)

dt = mrst_data["dt"]
if isa(dt, Real)
    dt = [dt]
end
timesteps = vec(dt)
res_context = model.context
w_context = DefaultContext()

initializer[:Reservoir] = init
forces[:Reservoir] = nothing
models[:Reservoir] = model

slw = 1.0
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
    pw[:Pressure] = Pressure(max_rel = 0.2)

    models[sym] = wi

    t_mrst = wdata["val"]
    is_injector = wdata["sign"] > 0
    is_shut = wdata["status"] < 1
    if wdata["type"] == "rate"
        target = SinglePhaseRateTarget(t_mrst, AqueousPhase())
    else
        target = BottomHolePressureTarget(t_mrst)
    end

    if is_shut
        println("Shut well")
        ctrl = DisabledControl()
    elseif is_injector
        ci = wdata["compi"]
        # ci = [0.5, 0.5]
        ctrl = InjectorControl(target, ci)
    else
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
#
g = WellGroup(well_symbols)
mode = PredictionMode()
WG = SimulationModel(g, mode)
F0 = Dict(:TotalSurfaceMassRate => 0.0)

facility_forces = build_forces(WG, control = controls)
models[:Facility] = WG
initializer[:Facility] = F0
# Reservoir as first group
groups = repeat([2], length(models))
groups[1] = 1

outer_context = DefaultContext()

red = :schur_apply
mmodel = MultiModel(convert_to_immutable_storage(models), groups = groups, context = outer_context, reduction = red)
# Set up joint state and simulate
state0 = setup_state(mmodel, initializer)
forces[:Facility] = facility_forces

parameters = setup_parameters(mmodel)
parameters[:Reservoir] = param_res

for w in well_symbols
    parameters[w] = well_parameters[w]
end 
dt = timesteps
# Set up linear solver and preconditioner
using AlgebraicMultigrid
p_solve = AMGPreconditioner(smoothed_aggregation)
cpr_type = :true_impes
update_interval = :once

prec = CPRPreconditioner(p_solve, strategy = cpr_type, 
                    update_interval = update_interval, partial_update = false)
atol = 1e-12
rtol = 0.005
max_it = 50

krylov = IterativeSolvers.gmres!
lsolve = GenericKrylov(krylov, verbose = 0, preconditioner = prec, 
                        relative_tolerance = rtol, absolute_tolerance = atol,
                        max_iterations = max_it)
m = 20
il = 1
dl = 0
# Simulate
sim = Simulator(mmodel, state0 = state0, parameters = deepcopy(parameters))
cfg = simulator_config(sim, info_level = il, debug_level = dl,
                            max_nonlinear_iterations = m,
                            output_states = true,
                            linear_solver = lsolve)
states, reports = simulate(sim, dt, forces = forces, config = cfg);
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