using Terv
ENV["JULIA_DEBUG"] = Terv

##
casename = "welltest"
G, mrst_data = get_minimal_tpfa_grid_from_mrst(casename, extraout = true)
## Set up reservoir part
# Parameters
bar = 1e5
p0 = 100*bar # 100 bar
mu = 1e-3    # 1 cP
cl = 1e-5/bar
pRef = 100*bar
rhoLS = 1000
rhoL = (p) -> rhoLS*exp((p - pRef)*cl)
phase = LiquidPhase()
sys = SinglePhaseSystem(phase)
model = SimulationModel(G, sys)

init = Dict(:Pressure => p0)
state0r = setup_state(model, init)

# Model parameters
param_res = setup_parameters(model)
param_res[:Viscosity] = [mu]
param_res[:Density] = [rhoL]

timesteps = [1.0]
sim = Simulator(model, state0 = state0r, parameters = param_res)
simulate(sim, timesteps)
## Set up injector
ix = 1

function build_well(mrst_data, ix)
    W_mrst = mrst_data["W"][ix]
    w = convert_to_immutable_storage(W_mrst)

    function awrap(x::Any)
        x
    end
    function awrap(x::Number)
        [x]
    end
    ref_depth = W_mrst["refDepth"]
    rc = Int64.(awrap(w.cells))
    n = length(rc)
    # dz = awrap(w.dZ)
    WI = awrap(w.WI)
    W = MultiSegmentWell(ones(n), rc, WI = WI, reference_depth = ref_depth)

    cell_centroids = copy((mrst_data["G"]["cells"]["centroids"])')
    z = vcat(ref_depth, cell_centroids[3, rc])
    flow = TwoPointPotentialFlow(SPU(), MixedWellSegmentFlow(), TotalMassVelocityMassFractionsFlow(), W, nothing, z)
    disc = (mass_flow = flow,)

    wmodel = SimulationModel(W, sys, discretization = disc)
    return wmodel
end

# Initial condition for all wells
w0 = Dict(:Pressure => p0, :TotalMassFlux => 0.0, :TotalWellMassRate => 0.0)

# Rate injector
Wi = build_well(mrst_data, 1)
ifrac = 0.01
irate = ifrac*sum(sum(G.grid.pore_volumes))/sum(timesteps)
it = SinglePhaseRateTarget(irate, phase)
ictrl = InjectorControl(it, 1.0)
istate = setup_state(Wi, w0)

# BHP producer
Wp = build_well(mrst_data, 2)
pt = BottomHolePressureTarget(pRef/2)
pctrl = ProducerControl(pt)
pstate = setup_state(Wp, w0)


##
mmodel = MultiModel((Reservoir = model, Injector = Wi, Producer = Wp))
# Set up joint state and simulate
state0 = setup_state(mmodel, Dict(:Reservoir => state0r, :Injector => istate, :Producer => pstate))
forces = Dict(:Reservoir => nothing, :Injector => ictrl, :Producer => pctrl)


parameters = setup_parameters(mmodel)
parameters[:Reservoir] = param_res
parameters[:Injector] = param_res
parameters[:Producer] = param_res


sim = Simulator(mmodel, state0 = state0, parameters = parameters)
states = simulate(sim, [1.0], forces = forces)

