using Terv
G = get_minimal_tpfa_grid_from_mrst("pico")
nc = number_of_cells(G)
nf = number_of_faces(G)


p0 = 100*1e5 # 100 bar
mu = 1e-3    # 1 cP

phase = LiquidPhase()
sys = SinglePhaseSystem(phase)


model = SimulationModel(G, sys)
storage = allocate_storage(model)

# System state
state0 = Dict()
state0["Pressure"] = repeat([100*1e5], nc) # 100

state = deepcopy(state0)
# Assign in primary variable
state["Pressure"] = allocate_vector_ad(state0["Pressure"], 1, diag_pos = 1)

storage["state0"] = state0
storage["state"] = state
# Model parameters
parameters = Dict()
parameters["Viscosity_L"] = mu

storage["parameters"] = parameters

update_equations!(model, storage)
# newton_step(model, storage)
