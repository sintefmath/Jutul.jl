using Terv

sys = ScalarTestSystem()
A = ScalarTestDomain()
B = ScalarTestDomain()
modelA = SimulationModel(A, sys)
modelB = SimulationModel(B, sys)


model = MultiModel((A = modelA, B = modelB))

parameters = setup_parameters(model)

# state0A = setup_state(modelA, Dict("XVar"=>1.0))
# state0B = setup_state(modelA, Dict("XVar"=>0.0))
state0 = setup_state(model, Dict("XVar"=>1.0), Dict("XVar"=>0.0))

sim = Simulator(model, state0 = state0, parameters = parameters)
states = simulate(sim, [1.0])
