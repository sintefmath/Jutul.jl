using Jutul
using Test

function test_io()
    sys = ScalarTestSystem()
    D = ScalarTestDomain()
    model = SimulationModel(D, sys)

    source = ScalarTestForce(1.0)
    forces = setup_forces(model, sources = source)
    state0 = setup_state(model, Dict(:XVar=>0.0))

    out = tempname()
    sim = Simulator(model, state0 = state0)
    states, reports = simulate(sim, [1.0, 2.0], forces = forces, output_states = true, output_path = out, info_level = -1)
    states2, reports2 = read_results(out)
    @testset "Test serialization of results" begin
        for (s_mem, s_file) in zip(states, states2)
            @test s_mem == s_file
        end
        for (r_mem, r_file) in zip(reports, reports2)
            for (k, v) in r_file
                @test v == r_mem[k]
            end
        end
    end
end

test_io()

function test_restart()
    sys = ScalarTestSystem()
    D = ScalarTestDomain()
    model = SimulationModel(D, sys)

    source = ScalarTestForce(1.0)
    forces = setup_forces(model, sources = source)
    state0 = setup_state(model, Dict(:XVar=>0.0))

    out = tempname()
    sim = Simulator(model, state0 = state0)
    states, reports = simulate(sim, [1.0, 2.0], forces = forces, output_states = true, output_path = out, info_level = -1)

    states2, reports2 = simulate(sim, [1.0, 2.0], forces = forces, restart = 2, output_path = out, info_level = -1)

    @testset "Test restart from stored results" begin
        for (s_mem, s_file) in zip(states, states2)
            @test s_mem == s_file
        end
    end
end

test_restart()

@testset "output_function" begin
    sys = ScalarTestSystem()
    D = ScalarTestDomain()
    model = SimulationModel(D, sys)

    source = ScalarTestForce(1.0)
    forces = setup_forces(model, sources = source)
    state0 = setup_state(model, Dict(:XVar=>0.0))
    sim = Simulator(model, state0 = state0)
    out = tempname()

    function update_test_function(state, report)
        state = deepcopy(state)
        state[:XVar] = state[:XVar] .+ 1.0
        if haskey(state, :substates)
            for substate in state[:substates]
                substate[:ExtraAddedField] = true
            end
        end
        return state
    end
    states, reports = simulate(sim, [1.0, 2.0], forces = forces, output_states = true, output_path = out, info_level = -1)

    states2, reports2 = simulate(sim, [1.0, 2.0], forces = forces,
        restart = false,
        output_path = out,
        info_level = -1,
        state0 = state0,
        output_function = update_test_function
    )

    states2_mem, reports2_mem = simulate(sim, [1.0, 2.0], forces = forces,
        restart = false,
        info_level = -1,
        state0 = state0,
        output_function = update_test_function
    )

    @testset "Test output function" begin
        for (s_mem, s_modif) in zip(states, states2)
            @test s_mem[:XVar] .+ 1.0 == s_modif[:XVar]
        end
        for (s_mem, s_modif) in zip(states, states2_mem)
            @test s_mem[:XVar] .+ 1.0 == s_modif[:XVar]
        end
    end

    states3, reports3 = simulate(sim, [1.0, 2.0], forces = forces,
        restart = false,
        output_path = out,
        info_level = -11,
        max_timestep = 0.5,
        output_substates = true,
        state0 = state0,
        output_function = update_test_function
    )

    @testset "Test output function" begin
        for (s_mem, s_modif) in zip(states, states3)
            @test s_mem[:XVar] .+ 1.0 == s_modif[:XVar]
            @test haskey(s_modif, :substates)
            for substate in s_modif[:substates]
                @test haskey(substate, :ExtraAddedField)
                @test substate[:ExtraAddedField] == true
            end
        end
    end
end

