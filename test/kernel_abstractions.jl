using Test
using Jutul
using KernelAbstractions
using SparseArrays

@testset "KernelAbstractions context" begin
    for use_manual in (true, false)
        model = SimulationModel(
            ScalarTestDomain(use_manual = use_manual),
            ScalarTestSystem(),
            context = ParallelCSRContext(1)
        )
        state0 = setup_state(model, Dict(:XVar => 0.0))
        cpu_simulator = Simulator(model, state0 = state0)
        simulator = transfer_to_backend(cpu_simulator, CPU())

        @test simulator.model.context isa KernelAbstractionsContext
        @test simulator.model.primary_variables isa NamedTuple
        @test simulator.storage.primary_variables.XVar === simulator.storage.state.XVar
        @test simulator.storage.LinearizedSystem.jac_buffer ===
            nonzeros(simulator.storage.LinearizedSystem.jac)
        @test cpu_simulator.storage.state.XVar isa Vector

        forces = setup_forces(model, sources = ScalarTestForce(1.0))
        states, = simulate!(simulator, [1.0], forces = forces, info_level = -1)
        @test only(states[end][:XVar]) ≈ 1.0
    end
end

@testset "KernelAbstractions generic AD stencil" begin
    grid = CartesianMesh((4, 4), (1.0, 1.0))
    model = SimulationModel(
        DiscretizedDomain(grid),
        SimpleHeatSystem(),
        context = ParallelCSRContext(1)
    )
    initial_temperature = collect(range(0.1, 1.0; length = number_of_cells(grid)))
    state0 = setup_state(model, Dict(:T => initial_temperature))

    reference_simulator = Simulator(model, state0 = state0)
    reference, = simulate!(reference_simulator, [0.1]; info_level = -1)

    cpu_simulator = Simulator(model, state0 = state0)
    simulator = transfer_to_backend(cpu_simulator, CPU())
    cache = simulator.storage.equations.heat_equation.Cells
    @test cache isa Jutul.GenericAutoDiffCache
    @test simulator.model.domain.representation.tags === nothing

    states, = simulate!(simulator, [0.1]; info_level = -1)
    @test states[end][:T] ≈ reference[end][:T] rtol = 1e-10
end

@testset "KernelAbstractions grouped multimodel" begin
    system = ScalarTestSystem()
    model_a = SimulationModel(ScalarTestDomain(), system)
    model_b = SimulationModel(ScalarTestDomain(), system)
    model = MultiModel((A = model_a, B = model_b), groups = [1, 2],
        group_execution = [SolveFullyOnDevice, AssembleOnDevice])
    add_cross_term!(model, ScalarTestCrossTerm();
        target = :A, source = :B, equation = :test_equation)

    state_a = setup_state(model_a, Dict(:XVar => 0.0))
    state_b = setup_state(model_b, Dict(:XVar => 0.0))
    state0 = setup_state(model; A = state_a, B = state_b)
    forces = setup_forces(model;
        A = setup_forces(model_a, sources = ScalarTestForce(1.0)),
        B = setup_forces(model_b, sources = ScalarTestForce(-1.0)))

    simulator = transfer_to_backend(
        Simulator(model; state0 = state0), CPU())
    @test collect(simulator.model.group_execution) ==
        [SolveFullyOnDevice, AssembleOnDevice]
    @test simulator.storage.host_evaluation.keys == (:B,)
    dt = 1.0
    Jutul.update_before_step!(simulator, dt, forces; time = 0.0)
    Jutul.update_state_dependents!(simulator.storage, simulator.model, dt, forces;
        time = dt)
    Jutul.update_linearized_system!(simulator.storage, simulator.model)

    linearized_system = simulator.storage.LinearizedSystem
    @test linearized_system isa Jutul.MultiLinearizedSystem
    @test all(isfinite, linearized_system.r_buffer)
    @test all(block -> all(isfinite, nonzeros(block.jac)),
        linearized_system.subsystems)
    @test all(cross_term ->
            cross_term.target_impact_map.entries isa AbstractVector,
        simulator.storage.cross_terms)

    cpu_only = MultiModel((A = model_a, B = model_b), groups = [1, 2],
        group_execution = NothingOnDevice)
    cpu_only_simulator = Simulator(cpu_only; state0 = state0)
    @test transfer_to_backend(cpu_only_simulator, CPU()) === cpu_only_simulator
end
