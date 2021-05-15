using Terv
ENV["JULIA_DEBUG"] = Terv

sys = ScalarTestSystem()
A = ScalarTestDomain()
B = ScalarTestDomain()
modelA = SimulationModel(A, sys)
modelB = SimulationModel(B, sys)

## Test the first model
sourceA = ScalarTestForce(1.0)
forcesA = build_forces(modelA, sources = sourceA)
state0A = setup_state(modelA, Dict("XVar"=>1.0))
sim = Simulator(modelA, state0 = state0A)
states = simulate(sim, [1.0], forces = forcesA)



## Make a multimodel
model = MultiModel((A = modelA, B = modelB))

parameters = setup_parameters(model)

##
# state0B = setup_state(modelA, Dict("XVar"=>0.0))
state0 = setup_state(model, Dict("XVar"=>1.0), Dict("XVar"=>0.0))
## Simulate
sim = Simulator(model, state0 = state0, parameters = parameters)
states = simulate(sim, [1.0])
