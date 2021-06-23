using Terv
ENV["JULIA_DEBUG"] = Terv
single_phase = false

##
casename = "welltest"
# casename = "2cell_well"
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

if single_phase
    sys = SinglePhaseSystem(phase)
    imix = 1.0
else
    phase2 = AqueousPhase()
    sys = ImmiscibleSystem([phase, phase2])
    imix = [1.0, 0.0]
end
model = SimulationModel(G, sys)

init = Dict(:Pressure => p0, :Saturations => [0, 1.0])
state0r = setup_state(model, init)

# timesteps = [1.0]
timesteps = [1.0, 1.0, 10.0, 10.0, 100.0]*3600*24

sim = Simulator(model, state0 = state0r)
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
w0 = Dict(:Pressure => p0, :TotalMassFlux => 1e-12, :Saturations => [0, 1.0])

# Rate injector
Wi = build_well(mrst_data, 1)
ifrac = 1.0
irate = ifrac*sum(sum(G.grid.pore_volumes))/sum(timesteps)
istate = setup_state(Wi, w0)
## Simulate injector on its own
sim_i = Simulator(Wi, state0 = istate)
states = simulate(sim_i, [1.0])
## Set up injector control
it = SinglePhaseRateTarget(irate, phase)
ictrl = InjectorControl(it, imix)
# iforces = build_forces(Wi, control = ictrl)

## BHP producer
Wp = build_well(mrst_data, 2)
pstate = setup_state(Wp, w0)
## Simulate producer only
sim_p = Simulator(Wp, state0 = pstate)
states = simulate(sim_p, [1.0])
## Set up producer control
pt = BottomHolePressureTarget(pRef/2)
pctrl = ProducerControl(pt)
# pforces = build_forces(Wp, control = pctrl)

##
g = WellGroup([:Injector, :Producer])
mode = PredictionMode()
WG = SimulationModel(g, mode)
F0 = Dict(:TotalSurfaceMassRate => 0.0)
fstate = setup_state(WG, F0)

controls = Dict(:Injector => ictrl, :Producer => pctrl)
fforces = build_forces(WG, control = controls)
##
mmodel = MultiModel((Reservoir = model, Injector = Wi, Producer = Wp, Facility = WG))
# Set up joint state and simulate
state0 = setup_state(mmodel, Dict(:Reservoir => state0r, :Injector => istate, :Producer => pstate, :Facility => fstate))
forces = Dict(:Reservoir => nothing, :Facility => fforces, :Injector => nothing, :Producer => nothing)


sim = Simulator(mmodel, state0 = state0)
states = simulate(sim, timesteps, forces = forces)
##
using GLMakie
f = Figure()

# Production/injection rates
q_i = map((x) -> x[:Facility][:TotalSurfaceMassRate][1], states)
ax = Axis(f[1, 1], title = "Injector rate")
lines!(ax, q_i)

q_p = map((x) -> x[:Facility][:TotalSurfaceMassRate][2], states)
ax = Axis(f[1, 2], title = "Producer rate")
lines!(ax, q_p)
# BHP plotting together
bh_i = map((x) -> x[:Injector][:Pressure][1], states)
bh_p = map((x) -> x[:Producer][:Pressure][1], states)

ax = Axis(f[2, 1], title = "Bottom hole pressure")
l1 = lines!(ax, bh_i)
l2 = lines!(ax, bh_p)
axislegend(ax, [l1, l2], ["Injector", "Producer"])

# Liquid fractions
if single_phase
    L_i = ones(size(bh_i))
    L_p = L_i
else
    L_i = map((x) -> x[:Injector][:Saturations][1], states)
    L_p = map((x) -> x[:Producer][:Saturations][1], states)
end

ax = Axis(f[2, 2], title = "Liquid volume fraction")
l1 = lines!(ax, L_i)
l2 = lines!(ax, L_p)
axislegend(ax, [l1, l2], ["Injector", "Producer"])
display(f)
