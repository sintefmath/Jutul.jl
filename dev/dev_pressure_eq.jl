using Terv
casename = "pico"
# casename = "spe10"
G = get_minimal_tpfa_grid_from_mrst(casename)
nc = number_of_cells(G)
nf = number_of_faces(G)

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
storage = allocate_storage(model)

# System state
p = repeat([p0], nc) # 100
src = sources = [SourceTerm(1, [1.0]), SourceTerm(nc, [-1.0])]

state0 = Dict()
state0["Pressure"] = p

state = deepcopy(state0)
# Assign in primary variable
pAD = allocate_vector_ad(state0["Pressure"], 1, diag_pos = 1)
state["Pressure"] = pAD

storage["state0"] = state0
storage["state"] = state
# Model parameters
parameters = Dict()
parameters["Viscosity_L"] = mu
parameters["Density_L"] = rhoL
storage["parameters"] = parameters

update_equations!(model, storage, dt = 1, sources = src)
update_linearized_system!(model, storage)

# newton_step(model, storage)
ref = Terv.get_incomp_matrix(G)

law = storage["ConservationLaw_L"]
lsys = storage["LinearizedSystem"]
r = lsys.r
jac = lsys.jac

dx = jac\r
dx/bar