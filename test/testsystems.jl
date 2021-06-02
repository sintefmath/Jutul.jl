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

@testset "Scalar test system" begin
    @test test_single()
end
