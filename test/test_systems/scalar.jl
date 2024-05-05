using Jutul
using Test

function test_single(use_manual)
    sys = ScalarTestSystem()
    D = ScalarTestDomain(use_manual = use_manual)
    model = SimulationModel(D, sys)
    source = ScalarTestForce(1.0)
    forces = setup_forces(model, sources = source)
    state0 = setup_state(model, Dict(:XVar=>0.0))
    sim = Simulator(model, state0 = state0)
    dt = [1.0]
    states, = simulate!(sim, dt, forces = forces, info_level = -1)
    @test length(states) == 1
    X = states[end][:XVar]
    @test only(X) ≈ 1.0

    states, reports = simulate(state0, model, dt, forces = forces, info_level = -1)
    @test length(states) == 1
    state = states[end]
    X = state[:XVar]
    @test only(X) ≈ 1.0
    @test !haskey(state, :substates)
    states, dt, = Jutul.expand_to_ministeps(states, reports)
    @test length(states) == 1

    states, reports = simulate(state0, model, dt,
        forces = forces,
        info_level = -1,
        max_timestep = 0.5,
        output_substates = true
    )
    state = states[end]
    @test haskey(state, :substates)
    @test only(only(state[:substates])[:XVar]) ≈ 0.5
    X = states[end][:XVar]
    @test only(X) ≈ 1.0

    states, dt, = Jutul.expand_to_ministeps(states, reports)
    @test length(states) == 2
end

@testset "Scalar test system" begin
    test_single(true)
    test_single(false)
end
