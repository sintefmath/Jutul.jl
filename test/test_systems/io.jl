using Jutul
using Test

function test_io()
    sys = ScalarTestSystem()
    D = ScalarTestDomain()
    model = SimulationModel(D, sys)
    
    source = ScalarTestForce(1.0)
    forces = build_forces(model, sources = source)
    state0 = setup_state(model, Dict(:XVar=>0.0))

    out = tempname()
    sim = Simulator(model, state0 = state0)
    states, reports = simulate(sim, [1.0, 2.0], forces = forces, output_states = true, output_path = out)
    states2, reports2 = read_results(out)
    @testset "Test serialization of results" begin
        for (s_mem, s_file) in zip(states, states2)
            @test s_mem == s_file
        end
        for (r_mem, r_file) in zip(reports, reports2)
            @test r_mem == r_file
        end
    end
end

test_io()

function test_restart()
    sys = ScalarTestSystem()
    D = ScalarTestDomain()
    model = SimulationModel(D, sys)
    
    source = ScalarTestForce(1.0)
    forces = build_forces(model, sources = source)
    state0 = setup_state(model, Dict(:XVar=>0.0))

    out = tempname()
    sim = Simulator(model, state0 = state0)
    states, reports = simulate(sim, [1.0, 2.0], forces = forces, output_states = true, output_path = out)

    states2, reports2 = simulate(sim, [1.0, 2.0], forces = forces, restart = 2, output_path = out)

    @testset "Test restart from stored results" begin
        for (s_mem, s_file) in zip(states, states2)
            @test s_mem == s_file
        end
    end
end

test_restart()
