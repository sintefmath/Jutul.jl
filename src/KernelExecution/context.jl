"""
    KernelAbstractionsContext(backend; float_type=Float64, index_type=Int,
                              linear_float_type=float_type,
                              linear_index_type=index_type,
                              matrix_layout=EquationMajorLayout(),
                              workgroupsize=256, minbatch=minbatch(nothing),
                              use_kernels_for_secondary=!is_cpu_backend,
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

`use_kernels_for_secondary` controls whether secondary properties use
KernelAbstractions kernels. It defaults to `false` for the CPU backend, where
the regular host-parallel path is generally preferable, and to `true` for
accelerator backends, which require kernel evaluation for device arrays.

`float_type` and `index_type` select assembly and state storage. The
`linear_float_type` and `linear_index_type` keywords select the linearized
system and solver storage, and default to the corresponding assembly types.

With `reduce_memory=true`, TPFA conservation laws without face-variable fluxes
use fused equation assembly, computing cell half-face flux values on the fly
instead of retaining them in backend storage.
"""
struct KernelAbstractionsContext{B, F, I, L, LF, LI} <: GPUJutulContext
    backend::B
    matrix_layout::L
    workgroupsize::Int
    minbatch::Int
    use_kernels_for_secondary::Bool
    reduce_memory::Bool
end

function KernelAbstractionsContext(backend;
        float_type::Type{F} = Float64,
        index_type::Type{I} = Int,
        linear_float_type::Type{LF} = float_type,
        linear_index_type::Type{LI} = index_type,
        use_kernels_for_secondary = missing,
        matrix_layout = EquationMajorLayout(),
        workgroupsize = 256,
        minbatch = 1000,
        reduce_memory = true
    ) where {F, I, LF, LI}
    if !(backend isa KernelAbstractions.Backend)
        throw(ArgumentError("backend must be a KernelAbstractions.Backend"))
    end
    if !(F <: AbstractFloat)
        throw(ArgumentError("float_type must be an AbstractFloat type"))
    end
    if !(I <: Integer) || I === Bool
        throw(ArgumentError("index_type must be a non-Bool Integer type"))
    end
    if !(LF <: AbstractFloat)
        throw(ArgumentError("linear_float_type must be an AbstractFloat type"))
    end
    if !(LI <: Integer) || LI === Bool
        throw(ArgumentError("linear_index_type must be a non-Bool Integer type"))
    end
    if workgroupsize <= 0
        throw(ArgumentError("workgroupsize must be positive"))
    end
    if minbatch <= 0
        throw(ArgumentError("minbatch must be positive"))
    end
    if ismissing(use_kernels_for_secondary)
        use_kernels_for_secondary = !(backend isa KernelAbstractions.CPU)
    end
    if !use_kernels_for_secondary &&
            !(backend isa KernelAbstractions.CPU)
        throw(ArgumentError(
            "use_kernels_for_secondary=false requires a CPU backend"))
    end
    return KernelAbstractionsContext{typeof(backend), F, I, typeof(matrix_layout), LF, LI}(
        backend, matrix_layout, Int(workgroupsize), Int(minbatch),
        use_kernels_for_secondary, reduce_memory
    )
end

float_type(::KernelAbstractionsContext{B, F}) where {B, F} = F
index_type(::KernelAbstractionsContext{B, F, I}) where {B, F, I} = I
linear_float_type(::KernelAbstractionsContext{B, F, I, L, LF}) where {B, F, I, L, LF} = LF
linear_index_type(::KernelAbstractionsContext{B, F, I, L, LF, LI}) where {B, F, I, L, LF, LI} = LI
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
secondary_variables_use_device_kernels(ctx::KernelAbstractionsContext) =
    ctx.use_kernels_for_secondary
function secondary_variables_thread_context(ctx::KernelAbstractionsContext)
    @assert is_cpu_backend(ctx)
    return :batch
end

function Base.adjoint(ctx::KernelAbstractionsContext)
    return KernelAbstractionsContext(ctx.backend;
        float_type = float_type(ctx),
        index_type = index_type(ctx),
        linear_float_type = linear_float_type(ctx),
        linear_index_type = linear_index_type(ctx),
        matrix_layout = adjoint(matrix_layout(ctx)),
        workgroupsize = ctx.workgroupsize,
        minbatch = minbatch(ctx),
        use_kernels_for_secondary = ctx.use_kernels_for_secondary,
        reduce_memory = ctx.reduce_memory
    )
end

function linear_solver_context(ctx::KernelAbstractionsContext)
    if linear_float_type(ctx) === float_type(ctx) &&
            linear_index_type(ctx) === index_type(ctx)
        return ctx
    end
    return KernelAbstractionsContext(ctx.backend;
        float_type = linear_float_type(ctx),
        index_type = linear_index_type(ctx),
        matrix_layout = matrix_layout(ctx),
        workgroupsize = ctx.workgroupsize,
        minbatch = minbatch(ctx),
        use_kernels_for_secondary = ctx.use_kernels_for_secondary,
        reduce_memory = ctx.reduce_memory)
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

function threaded_loop(f, n, ctx::KernelAbstractionsContext; do_wait = true)
    event = launch_threaded_loop(f, n, ctx)
    if !isnothing(event) && do_wait
        wait(event)
    end
    return nothing
end

function threaded_loop_minbatch(f, n, ctx::KernelAbstractionsContext,
        cpu_minbatch::Int = minbatch(ctx);
        do_wait::Bool = true
    )
    if !is_cpu_backend(ctx)
        return threaded_loop(f, n, ctx)
    end
    event = launch_threaded_loop(f, n, ctx; cpu_minbatch = cpu_minbatch)
    if !isnothing(event) && do_wait
        wait(event)
    end
    return nothing
end

@kernel function secondary_variable_update_kernel!(dest, var, model,
        @Const(dependencies))
    i = @index(Global)
    Jutul.update_secondary_variable!(dest, var, model, dependencies, i:i)
end

function secondary_variable_loop!(state, model, k::Symbol,
        ctx::KernelAbstractionsContext; do_wait = true)
    dest = state[k]
    n = length(Jutul.entity_eachindex(dest))
    if n == 0
        return nothing
    end
    var = model[k]
    deps = Tuple(Jutul.get_dependencies(var, model))
    dependencies = NamedTuple{deps}(ntuple(i -> state[deps[i]], length(deps)))
    kernel! = secondary_variable_update_kernel!(ctx.backend, ctx.workgroupsize)
    event = kernel!(dest, var, model, dependencies; ndrange = n)
    if !isnothing(event) && do_wait
        wait(event)
    end
    return event
end
