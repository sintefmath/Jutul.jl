"""
    KernelAbstractionsContext(backend; float_type=Float64, index_type=Int,
                              matrix_layout=EquationMajorLayout(), workgroupsize=256)

Execution context for a [`SimulationModel`](@ref) or [`MultiModel`](@ref) on a
KernelAbstractions backend. Build the model and [`Simulator`](@ref) on the CPU,
then use [`transfer_to_backend`](@ref) to adapt the completed simulator. A
multimodel application can select host evaluation, device assembly, or fully
host-side execution per linear-system group through
[`DeviceExecutionMode`](@ref).
"""
struct KernelAbstractionsContext{B, F, I, L} <: GPUJutulContext
    backend::B
    matrix_layout::L
    workgroupsize::Int
end

function KernelAbstractionsContext(backend;
        float_type::Type{F} = Float64,
        index_type::Type{I} = Int,
        matrix_layout = EquationMajorLayout(),
        workgroupsize = 256
    ) where {F, I}
    backend isa KernelAbstractions.Backend || throw(ArgumentError("backend must be a KernelAbstractions.Backend"))
    workgroupsize > 0 || throw(ArgumentError("workgroupsize must be positive"))
    return KernelAbstractionsContext{typeof(backend), F, I, typeof(matrix_layout)}(
        backend, matrix_layout, workgroupsize
    )
end

float_type(::KernelAbstractionsContext{B, F}) where {B, F} = F
index_type(::KernelAbstractionsContext{B, F, I}) where {B, F, I} = I
nzval_index_type(ctx::KernelAbstractionsContext) = index_type(ctx)
matrix_layout(ctx::KernelAbstractionsContext) = ctx.matrix_layout
nthreads(::KernelAbstractionsContext) = 1
minbatch(::KernelAbstractionsContext) = 1

function synchronize(ctx::KernelAbstractionsContext)
    if applicable(KernelAbstractions.synchronize, ctx.backend)
        KernelAbstractions.synchronize(ctx.backend)
    end
    return ctx
end

KernelAbstractions.get_backend(ctx::KernelAbstractionsContext) = ctx.backend

function Base.adjoint(ctx::KernelAbstractionsContext)
    return KernelAbstractionsContext(ctx.backend;
        float_type = float_type(ctx),
        index_type = index_type(ctx),
        matrix_layout = adjoint(matrix_layout(ctx)),
        workgroupsize = ctx.workgroupsize
    )
end


function threaded_loop(f, n, ctx::KernelAbstractionsContext)
    n <= 0 && return nothing
    @kernel function loop_kernel(F)
        i = @index(Global)
        F(i)
    end
    kernel! = loop_kernel(ctx.backend, ctx.workgroupsize)
    event = kernel!(f; ndrange = n)
    isnothing(event) || wait(event)
    return nothing
end

function threaded_loop_minbatch(f, n, ctx::KernelAbstractionsContext, minbatch::Int = 1)
    return threaded_loop(f, n, ctx)
end
