using Terv
using Test

function test_single()
    sys = ScalarTestSystem()
    D = ScalarTestDomain()
    model = SimulationModel(D, sys)
    
    source = ScalarTestForce(1.0)
    forces = build_forces(model, sources = source)
    state0 = setup_state(model, Dict(:XVar=>0.0))
    sim = Simulator(model, state0 = state0)
    states = simulate(sim, [1.0], forces = forces)

    X = states[end][:XVar]
    return X[] ≈ 1.0
end

function test_multi()
    sys = ScalarTestSystem()
    # Model A
    A = ScalarTestDomain()
    modelA = SimulationModel(A, sys)
    sourceA = ScalarTestForce(1.0)
    forcesA = build_forces(modelA, sources = sourceA)
    state0A = setup_state(modelA, Dict(:XVar=>0.0))
    # Model B
    B = ScalarTestDomain()
    modelB = SimulationModel(B, sys)
    sourceB = ScalarTestForce(-1.0)
    forcesB = build_forces(modelB, sources = sourceB)
    state0B = setup_state(modelB, Dict(:XVar=>0.0))
    
    # Make a multimodel
    model = MultiModel((A = modelA, B = modelB), groups = [1, 1])
    # Set up joint state and simulate
    state0 = setup_state(model, state0A, state0B)
    forces = Dict(:A => forcesA, :B => forcesB)
    sim = Simulator(model, state0 = state0)
    states = simulate(sim, [1.0], forces = forces)

    XA = states[end][:A][:XVar]
    XB = states[end][:B][:XVar]

    return XA[] ≈ 1/3 && XB[] ≈ -1/3
end

@testset "Scalar test system" begin
    @test test_single()
end
@testset "Multi-model" begin
    @test test_multi()
end
