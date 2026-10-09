using Jutul, KernelAbstractions, Test

# Model ROCBackend's asynchronous, event-free launch without requiring a GPU.
mutable struct EventlessTestBackend <: KernelAbstractions.Backend
    pending::Vector{Function}
    synchronizations::Int
end
EventlessTestBackend() = EventlessTestBackend(Function[], 0)

function KernelAbstractions.synchronize(backend::EventlessTestBackend)
    backend.synchronizations += 1
    while !isempty(backend.pending)
        popfirst!(backend.pending)()
    end
    return nothing
end

function Jutul.KernelExecution.launch_threaded_loop(
        f, n, ctx::KernelAbstractionsContext{EventlessTestBackend};
        cpu_minbatch::Int = minbatch(ctx)
    )
    if n > 0
        push!(ctx.backend.pending, () -> foreach(f, 1:n))
    end
    return nothing
end

struct EventlessTestVariable <: Jutul.ScalarVariable end
Jutul.get_dependencies(::EventlessTestVariable, model) = (:Input,)
function Jutul.update_secondary_variable!(
        dest, ::EventlessTestVariable, model, dependencies, ix
    )
    for i in ix
        dest[i] = dependencies.Input[i] + 1
    end
    return dest
end

struct EventlessSecondaryLaunch
    backend::EventlessTestBackend
end
Jutul.KernelExecution.secondary_variable_update_kernel!(
    backend::EventlessTestBackend, workgroupsize
) = EventlessSecondaryLaunch(backend)
function (kernel::EventlessSecondaryLaunch)(dest, var, model, dependencies; ndrange)
    push!(kernel.backend.pending, () -> Jutul.update_secondary_variable!(
        dest, var, model, dependencies, 1:ndrange
    ))
    return nothing
end

@testset "KA event-free launch synchronization" begin
    backend = EventlessTestBackend()
    ctx = KernelAbstractionsContext(backend)
    result = zeros(Int, 3)
    Jutul.threaded_loop(i -> (result[i] = i), 3, ctx)
    @test result == 1:3
    @test backend.synchronizations == 1

    for loop in (Jutul.threaded_loop, Jutul.threaded_loop_minbatch)
        fill!(result, 0)
        count = backend.synchronizations
        loop(i -> (result[i] = 2i), 3, ctx; do_wait = false)
        @test result == zeros(Int, 3)
        @test backend.synchronizations == count
        Jutul.synchronize(ctx)
        @test result == 2:2:6
        @test_throws ErrorException loop(i -> error("device failure"), 1, ctx)
    end
    count = backend.synchronizations
    Jutul.threaded_loop(identity, 0, ctx)
    @test backend.synchronizations == count

    model = (Output = EventlessTestVariable(),)
    state = (Output = zeros(Int, 3), Input = [1, 2, 3])
    Jutul.KernelExecution.secondary_variable_loop!(state, model, :Output, ctx)
    @test state.Output == [2, 3, 4]
    fill!(state.Output, 0)
    count = backend.synchronizations
    Jutul.KernelExecution.secondary_variable_loop!(
        state, model, :Output, ctx; do_wait = false
    )
    @test state.Output == zeros(Int, 3)
    @test backend.synchronizations == count
    Jutul.synchronize(ctx)
    @test state.Output == [2, 3, 4]
end
