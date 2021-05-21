using Terv
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
state0 = setup_state(model, init)

# Model parameters
parameters = setup_parameters(model)
parameters[:Viscosity] = [mu]
parameters[:Density] = [rhoL]

timesteps = [1.0]
sim = Simulator(model, state0 = state0, parameters = parameters)
simulate(sim, timesteps)
## Set up well part
ix = 1
W_mrst = mrst_data["W"][ix]
w = convert_to_immutable_storage(W_mrst)

function awrap(x::Any)
    x
end
function awrap(x::Number)
    [x]
end
rc = awrap(w.cells)
n = length(rc)
dz = awrap(w.dZ)
WI = awrap(w.WI)
W = MultiSegmentWell(ones(n), rc, dz = dz, WI = WI)
wmodel = SimulationModel(W, sys)

