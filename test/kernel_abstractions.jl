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
