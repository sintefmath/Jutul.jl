using Test
using Jutul
using KernelAbstractions
using SparseArrays

@testset "KernelAbstractions context" begin
    model = SimulationModel(
        ScalarTestDomain(),
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
