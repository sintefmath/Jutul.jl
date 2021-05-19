using Terv
ENV["JULIA_DEBUG"] = Terv
##
sys = ScalarTestSystem()
A = ScalarTestDomain()
B = ScalarTestDomain()
modelA = SimulationModel(A, sys)
modelB = SimulationModel(B, sys)

## Test the first model
println("Solving A")
sourceA = ScalarTestForce(1.0)
forcesA = build_forces(modelA, sources = sourceA)
state0A = setup_state(modelA, Dict(:XVar=>0.0))
simA = Simulator(modelA, state0 = state0A)
statesA = simulate(simA, [1.0], forces = forcesA)

## Test the second model
println("Solving B")

sourceB = ScalarTestForce(-1.0)
forcesB = build_forces(modelB, sources = sourceB)
state0B = setup_state(modelB, Dict(:XVar=>0.0))
simB = Simulator(modelB, state0 = state0B)
statesB = simulate(simB, [1.0], forces = forcesB)

## Make a multimodel
model = MultiModel((A = modelA, B = modelB), groups = [1, 1])
## Set up joint state and simulate
println("Solving A + B")
state0 = setup_state(model, state0A, state0B)
forces = Dict(:A => forcesA, :B => forcesB)
sim = Simulator(model, state0 = state0)
states = simulate(sim, [1.0], forces = forces)
