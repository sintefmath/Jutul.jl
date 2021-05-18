using Terv
ENV["JULIA_DEBUG"] = Terv
##
sys = ScalarTestSystem()
A = ScalarTestDomain()
B = ScalarTestDomain()
modelA = SimulationModel(A, sys)
modelB = SimulationModel(B, sys)

## Test the first model
sourceA = ScalarTestForce(1.0)
forcesA = build_forces(modelA, sources = sourceA)
state0A = setup_state(modelA, Dict(:XVar=>1.0))
simA = Simulator(modelA, state0 = state0A)
statesA = simulate(simA, [1.0], forces = forcesA)

## Test the second model
sourceB = ScalarTestForce(-1.0)
forcesB = build_forces(modelB, sources = sourceB)
state0B = setup_state(modelB, Dict(:XVar=>1.0))
simB = Simulator(modelB, state0 = state0B)
statesB = simulate(simB, [1.0], forces = forcesB)

## Make a multimodel
model = MultiModel((A = modelA, B = modelB), groups = [1, 1])

parameters = setup_parameters(model)

## Set up joint state and simulate
state0 = setup_state(model, Dict(:XVar=>1.0), Dict(:XVar=>0.0))
sim = Simulator(model, state0 = state0, parameters = parameters)
states = simulate(sim, [1.0])
