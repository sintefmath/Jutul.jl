using Test
using Jutul
using KernelAbstractions
using SparseArrays
using LinearAlgebra
import Adapt
import Jutul.KernelExecution: secondary_variable_evaluation_plan

@testset "KA CSR multiplication with zero beta" begin
    for T in (Float32, Float64)
        context = KernelAbstractionsContext(KernelAbstractions.CPU();
            float_type = T, index_type = Int32)
        @test Jutul.linear_float_type(context) === T
        @test Jutul.linear_index_type(context) === Int32
        matrix = sparse([1, 2], [1, 2], T[2, 3], 2, 2)
        csr = Adapt.adapt(context,
            Jutul.StaticSparsityMatrixCSR(copy(matrix')))
        result = fill(T(NaN), 2)
        mul!(result, csr, T[1, 2], one(T), zero(T))
        @test result == T[2, 6]
        mul!(result, csr, T[1, 2], one(T), one(T))
        @test result == T[4, 12]
        fill!(result, T(NaN))
        mul!(result, csr, T[1, 2], 1.0, 0.0)
        @test result == T[2, 6]
    end
    mixed_context = KernelAbstractionsContext(KernelAbstractions.CPU();
        float_type = Float32, index_type = Int32,
        linear_float_type = Float64, linear_index_type = Int64)
    @test Jutul.linear_float_type(mixed_context) === Float64
    @test Jutul.linear_index_type(mixed_context) === Int64
end

const mixed_cross_term_prepare_count = Ref(0)

mutable struct SynchronizationCountingContext <: Jutul.JutulContext
    count::Int
end

struct SynchronizationBoundaryModel{C} <: Jutul.JutulModel
    context::C
end

Jutul.synchronize(context::SynchronizationCountingContext) =
    (context.count += 1; context)
Jutul.matrix_layout(::SynchronizationCountingContext) = EquationMajorLayout()
Jutul.update_equations!(storage, model::SynchronizationBoundaryModel, dt;
    kwarg...) = nothing
Jutul.apply_forces!(storage, model::SynchronizationBoundaryModel, dt, forces;
    time = NaN, kwarg...) = nothing
Jutul.apply_boundary_conditions!(storage,
    model::SynchronizationBoundaryModel; kwarg...) = nothing

function Jutul.prepare_backend_transfer!(storage,
        model::SimulationModel{<:ScalarTestDomain})
    mixed_cross_term_prepare_count[] += 1
    return storage
end

struct KernelTransferParameter <: ScalarVariable end
Jutul.default_value(model, ::KernelTransferParameter) = 1.0

struct KernelArgumentTestAdaptor end
struct KernelArgumentArray{T}
    length::Int
end
Base.eltype(::KernelArgumentArray{T}) where T = T
Adapt.adapt_storage(::KernelArgumentTestAdaptor, array::AbstractArray) =
    KernelArgumentArray{eltype(array)}(length(array))

@testset "Evaluation synchronization boundaries" begin
    context = SynchronizationCountingContext(0)
    model = SynchronizationBoundaryModel(context)
    Jutul.update_equations_and_apply_forces!(
        nothing, model, 1.0, :force; do_sync = false)
    @test context.count == 0
    Jutul.update_equations_and_apply_forces!(
        nothing, model, 1.0, :force)
    @test context.count == 1

    multimodel = MultiModel((A = model,); context = context)
    storage = JutulStorage(cross_terms = Any[])
    Jutul.update_cross_terms!(storage, multimodel, 1.0; do_sync = false)
    @test context.count == 1
    Jutul.update_cross_terms!(storage, multimodel, 1.0)
    @test context.count == 2
end

@testset "ImmutableJutulStorage kernel arguments" begin
    mutable_storage = JutulStorage(values = (1.0, 2.0))
    other_mutable_storage = JutulStorage(other = "value")
    storage = convert_to_immutable_storage(mutable_storage)
    @test mutable_storage isa JutulStorage
    @test typeof(mutable_storage) === typeof(other_mutable_storage)
    @test JutulStorage(mutable_storage) === mutable_storage
    @test storage isa ImmutableJutulStorage
    @test isbitstype(typeof(storage))
    @test sizeof(storage) == sizeof(Jutul.data(storage))
    @test @inferred(getproperty(storage, :values)) === (1.0, 2.0)
    @test !isbitstype(typeof(mutable_storage))

    array_storage = ImmutableJutulStorage((values = [1.0, 2.0],))
    adapted = Adapt.adapt(KernelArgumentTestAdaptor(), array_storage)
    @test adapted isa ImmutableJutulStorage
    @test adapted.values isa KernelArgumentArray{Float64}
    @test isbitstype(typeof(adapted))
end

@testset "Kernel argument interpolation adaptation" begin
    interpolant = Jutul.BilinearInterpolant(
        [0.0, 1.0], [0.0, 1.0], [1.0 2.0; 3.0 4.0])
    kernel_interpolant = Adapt.adapt(
        KernelArgumentTestAdaptor(), interpolant)
    @test kernel_interpolant.X isa KernelArgumentArray{Float64}
    @test kernel_interpolant.Y isa KernelArgumentArray{Float64}
    @test kernel_interpolant.F isa KernelArgumentArray{Float64}
    @test isbitstype(typeof(kernel_interpolant))
end

struct SecondaryPlanA end
struct SecondaryPlanB end
struct SecondaryPlanC end
struct SecondaryPlanD end

struct SecondaryPlanTestModel{C, P, S, R}
    context::C
    primary_variables::P
    secondary_variables::S
    parameters::R
end

Jutul.get_dependencies(::SecondaryPlanA, model::SecondaryPlanTestModel) = (:X,)
Jutul.get_dependencies(::SecondaryPlanB, model::SecondaryPlanTestModel) = (:A,)
Jutul.get_dependencies(::SecondaryPlanC, model::SecondaryPlanTestModel) = (:X,)
Jutul.get_dependencies(::SecondaryPlanD, model::SecondaryPlanTestModel) = (:B, :C)
Jutul.number_of_entities(model::SecondaryPlanTestModel, ::SecondaryPlanA) = 7
Jutul.number_of_entities(model::SecondaryPlanTestModel, ::SecondaryPlanB) = 7
Jutul.number_of_entities(model::SecondaryPlanTestModel, ::SecondaryPlanC) = 3
Jutul.number_of_entities(model::SecondaryPlanTestModel, ::SecondaryPlanD) = 3

function Jutul.update_secondary_variable!(target, ::SecondaryPlanA,
        model::SecondaryPlanTestModel, state, ix)
    for i in ix
        target[i] = state.X[i] + 1
    end
end

function Jutul.update_secondary_variable!(target, ::SecondaryPlanB,
        model::SecondaryPlanTestModel, state, ix)
    for i in ix
        target[i] = 2*state.A[i]
    end
end

function Jutul.update_secondary_variable!(target, ::SecondaryPlanC,
        model::SecondaryPlanTestModel, state, ix)
    for i in ix
        target[i] = state.X[i] - 1
    end
end

function Jutul.update_secondary_variable!(target, ::SecondaryPlanD,
        model::SecondaryPlanTestModel, state, ix)
    for i in ix
        target[i] = state.B[i] + state.C[i]
    end
end

@testset "Secondary variable dependency levels" begin
    variables = (B = SecondaryPlanB(), A = SecondaryPlanA(),
        D = SecondaryPlanD(), C = SecondaryPlanC())
    model = SecondaryPlanTestModel(
        KernelAbstractionsContext(CPU();
            use_kernels_for_secondary = true, workgroupsize = 4),
        (X = nothing,), variables, NamedTuple())
    plan = secondary_variable_evaluation_plan(model)
    @test plan isa Vector{Vector{Pair{Symbol, Int}}}
    @test plan == [[:A => 7, :C => 3], [:B => 7], [:D => 3]]

    empty_model = SecondaryPlanTestModel(
        model.context, model.primary_variables, NamedTuple(), NamedTuple())
    empty_plan = secondary_variable_evaluation_plan(empty_model)
    @test empty_plan isa Vector{Vector{Pair{Symbol, Int}}}
    @test isempty(empty_plan)

    state = (
        X = collect(1.0:7.0),
        A = zeros(7),
        B = zeros(7),
        C = zeros(3),
        D = zeros(3)
    )
    Jutul.update_secondary_variables_state!(
        state, model, variables, plan)
    @test state.A == state.X .+ 1
    @test state.B == 2 .* state.A
    @test state.C == state.X[1:3] .- 1
    @test state.D == state.B[1:3] .+ state.C

    host_variables = (A = SecondaryPlanA(), C = SecondaryPlanC(),
        B = SecondaryPlanB(), D = SecondaryPlanD())
    host_model = SecondaryPlanTestModel(
        KernelAbstractionsContext(CPU(); minbatch = 1),
        (X = nothing,), host_variables, NamedTuple())
    host_state = (
        X = collect(1.0:7.0),
        A = zeros(7),
        B = zeros(7),
        C = zeros(3),
        D = zeros(3)
    )
    Jutul.update_secondary_variables_state!(
        host_state, host_model, host_variables)
    @test host_state.A == host_state.X .+ 1
    @test host_state.B == 2 .* host_state.A
    @test host_state.C == host_state.X[1:3] .- 1
    @test host_state.D == host_state.B[1:3] .+ host_state.C
end

@testset "KernelAbstractions context" begin
    threshold_context = KernelAbstractionsContext(CPU();
        minbatch = 4, workgroupsize = 2)
    @test threshold_context.reduce_memory
    @test !threshold_context.use_kernels_for_secondary
    @test !threshold_context.secondary_async
    @test minbatch(threshold_context) == 4
    @test minbatch(adjoint(threshold_context)) == 4
    @test !adjoint(threshold_context).use_kernels_for_secondary
    @test adjoint(threshold_context).reduce_memory
    kernel_context = KernelAbstractionsContext(CPU();
        use_kernels_for_secondary = true)
    @test kernel_context.use_kernels_for_secondary
    @test kernel_context.secondary_async
    @test adjoint(kernel_context).use_kernels_for_secondary
    @test !KernelAbstractionsContext(CPU(); reduce_memory = false).reduce_memory
    @test_throws ArgumentError KernelAbstractionsContext(CPU(); minbatch = 0)
    compatibility_context = KernelAbstractionsContext(CPU();
        secondary_async = true)
    @test compatibility_context.use_kernels_for_secondary
    @test compatibility_context.secondary_async

    small_result = zeros(Int, 4)
    small_event = Jutul.KernelExecution.launch_threaded_loop(
        i -> (small_result[i] = i), length(small_result), threshold_context)
    @test isnothing(small_event)
    @test small_result == 1:4

    for use_manual in (true, false)
        model = SimulationModel(
            ScalarTestDomain(use_manual = use_manual),
            ScalarTestSystem(),
            context = ParallelCSRContext(1)
        )
        state0 = setup_state(model, Dict(:XVar => 0.0))
        cpu_simulator = Simulator(model, state0 = state0)
        @test !haskey(cpu_simulator.storage.variable_definitions,
            :secondary_variable_evaluation_plan)
        simulator = transfer_to_backend(cpu_simulator, CPU())

        @test simulator.model.context isa KernelAbstractionsContext
        @test !haskey(simulator.storage, :evaluation_state)
        @test evaluation_state(simulator.storage) === simulator.storage.state
        @test evaluation_state0(simulator.storage) === simulator.storage.state0
        @test !haskey(simulator.storage.variable_definitions,
            :secondary_variable_evaluation_plan)
        @test minbatch(simulator.storage.LinearizedSystem.jac) ==
            minbatch(simulator.model.context)
        @test simulator.model.primary_variables isa NamedTuple
        @test simulator.storage.primary_variables.XVar === simulator.storage.state.XVar
        @test simulator.storage.LinearizedSystem.jac_buffer ===
            nonzeros(simulator.storage.LinearizedSystem.jac)
        @test cpu_simulator.storage.state.XVar isa Vector
        if use_manual
            cache = simulator.storage.equations.test_equation
            @test cache isa Jutul.CompactAutoDiffCache
            kernel_cache = Adapt.adapt(KernelArgumentTestAdaptor(), cache)
            @test kernel_cache.entries isa KernelArgumentArray
            @test kernel_cache.jacobian_positions isa KernelArgumentArray
            @test isbitstype(typeof(kernel_cache))
        end

        if use_manual
            kernel_simulator = transfer_to_backend(cpu_simulator,
                KernelAbstractionsContext(CPU();
                    use_kernels_for_secondary = true))
            @test haskey(kernel_simulator.storage.variable_definitions,
                :secondary_variable_evaluation_plan)
        end

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
    kernel_cache = Adapt.adapt(KernelArgumentTestAdaptor(), cache)
    @test kernel_cache.entries isa KernelArgumentArray
    @test kernel_cache.vpos isa KernelArgumentArray
    @test kernel_cache.variables isa KernelArgumentArray
    @test kernel_cache.jacobian_positions isa KernelArgumentArray
    @test isbitstype(typeof(kernel_cache))
    # Make sure tags do not survive...
    @test isnothing(simulator.model.domain.representation.tags)

    states, = simulate!(simulator, [0.1]; info_level = -1)
    @test states[end][:T] ≈ reference[end][:T] rtol = 1e-10
end

@testset "KernelAbstractions grouped multimodel" begin
    system = ScalarTestSystem()
    model_a = SimulationModel(ScalarTestDomain(), system)
    model_b = SimulationModel(ScalarTestDomain(), system)
    set_parameters!(model_b;
        KernelTransferParameter = KernelTransferParameter())
    model = MultiModel((A = model_a, B = model_b))
    add_cross_term!(model, ScalarTestCrossTerm();
        target = :A, source = :B, equation = :test_equation)

    state_a = setup_state(model_a, Dict(:XVar => 0.0))
    state_b = setup_state(model_b, Dict(:XVar => 0.0))
    state0 = setup_state(model; A = state_a, B = state_b)
    forces = setup_forces(model;
        A = setup_forces(model_a, sources = ScalarTestForce(1.0)),
        B = setup_forces(model_b, sources = ScalarTestForce(-1.0)))

    function group_execution(key, submodel)
        if key == :A
            return SolveFullyOnDevice
        else
            return AssembleOnDevice
        end
    end
    simulator = transfer_to_backend(
        Simulator(model; state0 = state0), CPU();
        group_execution = group_execution)
    @test isnothing(simulator.model.groups)
    @test collect(simulator.model.group_execution) ==
        [SolveFullyOnDevice, AssembleOnDevice]
    @test simulator.model[:A].context.reduce_memory
    @test !simulator.model[:B].context.reduce_memory
    @test simulator.storage.host_evaluation.keys == (:B,)
    host = simulator.storage.host_evaluation
    @test host.model[:A] !== simulator.model[:A]
    @test host.storage[:A] !== simulator.storage[:A]
    @test host.storage.state[:A] !== simulator.storage.state[:A]
    @test host.storage.state0[:A] !== simulator.storage.state0[:A]
    @test host.model[:B] !== simulator.model[:B]
    @test host.storage[:B] !== simulator.storage[:B]
    @test host.storage.cross_terms[1] !== simulator.storage.cross_terms[1]
    @test host.cross_term_evaluation.host == [1]
    @test host.cross_term_evaluation.mixed == [1]
    @test host.cross_term_evaluation.mixed_models == [:A]
    @test host.cross_term_evaluation.mixed_on_host

    # Mixed cross terms execute on the host by default. Only the current state
    # of the SolveFullyOnDevice model is copied back; state0 and parameters use
    # their retained CPU values.
    host.storage.B.state.XVar .= 2.0
    simulator.storage.B.state.XVar .= -10.0
    simulator.storage.A.state.XVar .= 5.0
    host.storage.A.state.XVar .= -5.0
    host.storage.A.state0.XVar .= 8.0
    simulator.storage.A.state0.XVar .= -8.0
    Jutul.maybe_synchronize_device_host!(simulator.storage, simulator.model;
        state = true, state0 = false, parameters = false)
    Jutul.update_cross_terms!(simulator.storage, simulator.model, 1.0)
    @test only(simulator.storage.B.state.XVar) == -10.0
    @test only(host.storage.A.state.XVar) == 5.0
    @test only(host.storage.A.state0.XVar) == 8.0
    mixed_entries = host.storage.cross_terms[1].target.Cells.entries
    @test Jutul.value(only(mixed_entries)) == 3.0
    Jutul.transfer_cross_term_evaluation!(simulator.storage, simulator.model)
    backend_entries = simulator.storage.cross_terms[1].target.Cells.entries
    @test Jutul.value(only(backend_entries)) == 3.0

    # The previous backend-evaluation strategy remains available explicitly.
    device_mixed_simulator = transfer_to_backend(
        Simulator(model; state0 = state0), CPU();
        group_execution = group_execution,
        mixed_cross_terms_on_host = false)
    device_host = device_mixed_simulator.storage.host_evaluation
    @test isempty(device_host.cross_term_evaluation.host)
    @test device_host.cross_term_evaluation.mixed == [1]
    @test device_host.cross_term_evaluation.mixed_models == [:B]
    @test !device_host.cross_term_evaluation.mixed_on_host
    device_host.storage.B.state.XVar .= 2.0
    device_mixed_simulator.storage.B.state.XVar .= -10.0
    device_mixed_simulator.storage.A.state.XVar .= 5.0
    device_host.storage.B.state0.XVar .= 8.0
    device_mixed_simulator.storage.B.state0.XVar .= -8.0
    device_host.storage.B.parameters.KernelTransferParameter .= 9.0
    device_mixed_simulator.storage.B.parameters.KernelTransferParameter .= -2.0
    device_mixed_simulator.storage.B.state0.KernelTransferParameter .= -3.0
    Jutul.maybe_synchronize_device_host!(
        device_mixed_simulator.storage, device_mixed_simulator.model;
        state = true, state0 = false, parameters = false)
    Jutul.update_cross_terms!(device_mixed_simulator.storage,
        device_mixed_simulator.model, 1.0)
    @test only(device_mixed_simulator.storage.B.state.XVar) == 2.0
    @test only(device_mixed_simulator.storage.B.state0.XVar) == -8.0
    @test only(device_mixed_simulator.storage.B.parameters.KernelTransferParameter) == -3.0
    device_entries =
        device_mixed_simulator.storage.cross_terms[1].target.Cells.entries
    @test Jutul.value(only(device_entries)) == 3.0
    Jutul.maybe_synchronize_device_host!(
        device_mixed_simulator.storage, device_mixed_simulator.model;
        state = false, state0 = true, parameters = false)
    @test only(device_mixed_simulator.storage.B.state0.XVar) == 8.0
    Jutul.maybe_synchronize_device_host!(
        device_mixed_simulator.storage, device_mixed_simulator.model;
        state = false, state0 = false, parameters = true)
    @test only(device_mixed_simulator.storage.B.parameters.KernelTransferParameter) == 9.0
    @test only(device_mixed_simulator.storage.B.state0.KernelTransferParameter) == 9.0

    host.storage.B.state.XVar .= 0.0
    host.storage.B.state0.XVar .= 0.0
    host.storage.B.parameters.KernelTransferParameter .= 1.0
    Jutul.maybe_synchronize_device_host!(simulator.storage, simulator.model)
    simulator.storage.B.state.XVar .= 0.0
    simulator.storage.A.state.XVar .= 0.0

    reverse_mixed = MultiModel((A = model_a, B = model_b))
    add_cross_term!(reverse_mixed, ScalarTestCrossTerm();
        target = :B, source = :A, equation = :test_equation)
    reverse_mixed_simulator = transfer_to_backend(
        Simulator(reverse_mixed; state0 = state0), CPU();
        group_execution = group_execution)
    reverse_host = reverse_mixed_simulator.storage.host_evaluation
    reverse_host.storage.B.state.XVar .= 2.0
    reverse_mixed_simulator.storage.A.state.XVar .= 5.0
    Jutul.maybe_synchronize_device_host!(
        reverse_mixed_simulator.storage, reverse_mixed_simulator.model)
    Jutul.update_cross_terms!(reverse_mixed_simulator.storage,
        reverse_mixed_simulator.model, 1.0)
    Jutul.transfer_cross_term_evaluation!(
        reverse_mixed_simulator.storage, reverse_mixed_simulator.model)
    reverse_entries =
        reverse_mixed_simulator.storage.cross_terms[1].target.Cells.entries
    @test Jutul.value(only(reverse_entries)) == -3.0

    overlapping_mixed = MultiModel((A = model_a, B = model_b))
    for _ in 1:2
        add_cross_term!(overlapping_mixed, ScalarTestCrossTerm();
            target = :A, source = :B, equation = :test_equation)
    end
    overlapping_mixed_simulator = transfer_to_backend(
        Simulator(overlapping_mixed; state0 = state0), CPU();
        group_execution = group_execution)
    overlapping_host = overlapping_mixed_simulator.storage.host_evaluation
    @test overlapping_host.cross_term_evaluation.host == [1, 2]
    @test overlapping_host.cross_term_evaluation.mixed == [1, 2]
    @test overlapping_host.cross_term_evaluation.mixed_models == [:A]
    mixed_cross_term_prepare_count[] = 0
    Jutul.update_equations_and_apply_forces!(
        overlapping_mixed_simulator.storage,
        overlapping_mixed_simulator.model, 1.0, forces;
        do_sync = false)
    Jutul.synchronize(overlapping_mixed_simulator.model.context)
    @test mixed_cross_term_prepare_count[] == 1

    adjoint_source = MultiModel((A = model_a, B = model_b), groups = [1, 2],
        group_execution = [SolveFullyOnDevice, AssembleOnDevice])
    adjoint_model = Jutul.adjoint_model_copy(
        adjoint_source;
        context = DefaultContext()
    )
    @test isnothing(adjoint_model.groups)
    @test all(==(SolveFullyOnDevice), adjoint_model.group_execution)

    @test_throws ArgumentError MultiModel((A = model_a, B = model_b),
        group_execution = [NothingOnDevice, SolveFullyOnDevice])
    mixed_groups = MultiModel((A = model_a, B = model_b), groups = [1, 2],
        group_execution = [NothingOnDevice, SolveFullyOnDevice])
    @test mixed_groups.groups == [1, 2]

    dt = 1.0
    Jutul.update_before_step!(simulator, dt, forces; time = 0.0)
    Jutul.update_state_dependents!(simulator.storage, simulator.model, dt, forces;
        time = dt)
    Jutul.update_linearized_system!(simulator.storage, simulator.model)

    linearized_system = simulator.storage.LinearizedSystem
    @test linearized_system isa Jutul.LinearizedSystem
    @test all(isfinite, linearized_system.r_buffer)
    @test all(isfinite, nonzeros(linearized_system.jac))
    @test all(cross_term ->
            cross_term.target_impact_map.entries isa AbstractVector,
        simulator.storage.cross_terms)

    cpu_only = MultiModel((A = model_a, B = model_b), groups = [1, 2],
        group_execution = NothingOnDevice)
    cpu_only_simulator = Simulator(cpu_only; state0 = state0)
    @test transfer_to_backend(cpu_only_simulator, CPU()) === cpu_only_simulator

    host_only = MultiModel((A = model_a, B = model_b),
        group_execution = AssembleOnDevice)
    add_cross_term!(host_only, ScalarTestCrossTerm();
        target = :A, source = :B, equation = :test_equation)
    host_only_simulator = transfer_to_backend(
        Simulator(host_only; state0 = state0), CPU())
    host_only_storage = host_only_simulator.storage.host_evaluation
    @test host_only_storage.keys == (:A, :B)
    @test host_only_storage.cross_term_evaluation.host == [1]
    @test isempty(host_only_storage.cross_term_evaluation.mixed)
    @test isempty(host_only_storage.cross_term_evaluation.mixed_models)
    @test host_only_storage.storage.cross_terms[1] !==
        host_only_simulator.storage.cross_terms[1]
    host_only_storage.storage.A.state.XVar .= 7.0
    host_only_storage.storage.B.state.XVar .= 4.0
    Jutul.maybe_synchronize_device_host!(
        host_only_simulator.storage, host_only_simulator.model)
    Jutul.update_cross_terms!(host_only_simulator.storage,
        host_only_simulator.model, 1.0)
    Jutul.transfer_cross_term_evaluation!(
        host_only_simulator.storage, host_only_simulator.model)
    host_entries = host_only_storage.storage.cross_terms[1].target.Cells.entries
    device_entries = host_only_simulator.storage.cross_terms[1].target.Cells.entries
    @test Jutul.value(only(host_entries)) == 3.0
    @test Jutul.value(only(device_entries)) == 3.0

    device_only = MultiModel((A = model_a, B = model_b),
        group_execution = SolveFullyOnDevice)
    add_cross_term!(device_only, ScalarTestCrossTerm();
        target = :A, source = :B, equation = :test_equation)
    device_only_simulator = transfer_to_backend(
        Simulator(device_only; state0 = state0), CPU())
    @test !haskey(device_only_simulator.storage, :host_evaluation)
    device_only_simulator.storage.A.state.XVar .= 7.0
    device_only_simulator.storage.B.state.XVar .= 4.0
    Jutul.update_cross_terms!(device_only_simulator.storage,
        device_only_simulator.model, 1.0)
    device_only_entries =
        device_only_simulator.storage.cross_terms[1].target.Cells.entries
    @test Jutul.value(only(device_only_entries)) == 3.0
end
