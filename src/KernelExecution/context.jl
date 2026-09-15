"""
    KernelAbstractionsContext(backend; float_type=Float64, index_type=Int,
                              matrix_layout=EquationMajorLayout(),
                              workgroupsize=256, minbatch=minbatch(nothing),
                              reduce_memory=true)

Execution context for a [`SimulationModel`](@ref) or [`MultiModel`](@ref) on a
KernelAbstractions backend. Build the model and [`Simulator`](@ref) on the CPU,
then use [`transfer_to_backend`](@ref) to adapt the completed simulator. A
multimodel application can select host evaluation, device assembly, or fully
host-side execution per submodel through
`DeviceExecutionMode`.

`minbatch` is the smallest CPU loop that is worth submitting to
KernelAbstractions. Smaller CPU loops execute directly on the calling thread;
device backends always launch kernels.

With `reduce_memory=true`, TPFA conservation laws without face-variable fluxes
use fused equation assembly, computing cell half-face flux values on the fly
instead of retaining them in backend storage.
"""
struct KernelAbstractionsContext{B, F, I, L} <: GPUJutulContext
    backend::B
    matrix_layout::L
    workgroupsize::Int
    minbatch::Int
    secondary_async::Bool
    reduce_memory::Bool
end

function KernelAbstractionsContext(backend;
        float_type::Type{F} = Float64,
        index_type::Type{I} = Int,
        secondary_async = !(backend isa KernelAbstractions.CPU),
        matrix_layout = EquationMajorLayout(),
        workgroupsize = 256,
        minbatch = 1000,
        reduce_memory = true
    ) where {F, I}
    if !(backend isa KernelAbstractions.Backend)
        throw(ArgumentError("backend must be a KernelAbstractions.Backend"))
    end
    if !(F <: AbstractFloat)
        throw(ArgumentError("float_type must be an AbstractFloat type"))
    end
    if !(I <: Integer) || I === Bool
        throw(ArgumentError("index_type must be a non-Bool Integer type"))
    end
    if workgroupsize <= 0
        throw(ArgumentError("workgroupsize must be positive"))
    end
    if minbatch <= 0
        throw(ArgumentError("minbatch must be positive"))
    end
    return KernelAbstractionsContext{typeof(backend), F, I, typeof(matrix_layout)}(
        backend, matrix_layout, Int(workgroupsize), Int(minbatch),
        secondary_async, reduce_memory
    )
end

float_type(::KernelAbstractionsContext{B, F}) where {B, F} = F
index_type(::KernelAbstractionsContext{B, F, I}) where {B, F, I} = I
nzval_index_type(ctx::KernelAbstractionsContext) = index_type(ctx)
matrix_layout(ctx::KernelAbstractionsContext) = ctx.matrix_layout
nthreads(::KernelAbstractionsContext) = 1
minbatch(ctx::KernelAbstractionsContext) = ctx.minbatch

function synchronize(ctx::KernelAbstractionsContext)
    if applicable(KernelAbstractions.synchronize, ctx.backend)
        KernelAbstractions.synchronize(ctx.backend)
    end
    return ctx
end

KernelAbstractions.get_backend(ctx::KernelAbstractionsContext) = ctx.backend
is_cpu_backend(ctx::KernelAbstractionsContext) =
    ctx.backend isa KernelAbstractions.CPU

function Base.adjoint(ctx::KernelAbstractionsContext)
    return KernelAbstractionsContext(ctx.backend;
        float_type = float_type(ctx),
        index_type = index_type(ctx),
        matrix_layout = adjoint(matrix_layout(ctx)),
        workgroupsize = ctx.workgroupsize,
        minbatch = minbatch(ctx),
        reduce_memory = ctx.reduce_memory
    )
end


@kernel function threaded_loop_kernel(f)
    index = @index(Global)
    f(index)
end

function launch_threaded_loop(f, n, ctx::KernelAbstractionsContext;
        cpu_minbatch::Int = minbatch(ctx))
    if n <= 0
        return nothing
    end
    if is_cpu_backend(ctx) && n <= cpu_minbatch
        @inbounds for index in 1:n
            f(index)
        end
        return nothing
    end
    kernel! = threaded_loop_kernel(ctx.backend, ctx.workgroupsize)
    return kernel!(f; ndrange = n)
end

function threaded_loop(f, n, ctx::KernelAbstractionsContext)
    event = launch_threaded_loop(f, n, ctx)
    if !isnothing(event)
        wait(event)
    end
    return nothing
end

function threaded_loop_minbatch(f, n, ctx::KernelAbstractionsContext,
        cpu_minbatch::Int = minbatch(ctx))
    if !is_cpu_backend(ctx)
        return threaded_loop(f, n, ctx)
    end
    event = launch_threaded_loop(f, n, ctx; cpu_minbatch = cpu_minbatch)
    if !isnothing(event)
        wait(event)
    end
    return nothing
end
