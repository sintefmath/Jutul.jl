module JutulCUDAExt

using Jutul
using Jutul.KAPreconditioners
using Jutul: StaticSparsityMatrixCSR
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixBSR, CuSparseMatrixCSR
using KernelAbstractions
using LinearAlgebra
using StaticArrays: StaticMatrix
import Adapt
import CUDA: KernelAdaptor
import KernelAbstractions as KA

struct CUDASmootherKernel{K, C, KC}
    kernel::K
    compiled::C
    calls::Vector{Union{Nothing, KC}}
    threads::Vector{Int}
    blocks::Vector{Int}
end

function cuda_smoother_launch_config(kernel, ndrange)
    ndrange, _, iterspace, _ = KA.launch_config(kernel, ndrange, nothing)
    context = KA.mkcontext(kernel, ndrange, iterspace)
    threads = length(KA.workitems(iterspace))
    blocks = length(KA.blocks(iterspace))
    return context, threads, blocks
end

function cuda_smoother_maxthreads(kernel)
    return if KA.workgroupsize(kernel) <: KA.StaticSize
        prod(KA.get(KA.workgroupsize(kernel)))
    else
        nothing
    end
end

function KAPreconditioners.setup_smoother_kernel(
        kernel, backend::CUDA.CUDABackend, block_size, arguments...;
        ndrange, number_of_launches = 1
    )
    kernel = kernel(backend, block_size)
    context, first_threads, first_blocks = cuda_smoother_launch_config(
        kernel, ndrange
    )
    call = CUDA.KernelCall(kernel.f, context, arguments...)
    maxthreads = cuda_smoother_maxthreads(kernel)
    compiled = CUDA.kernel_compile(
        call; always_inline = backend.always_inline, maxthreads = maxthreads
    )
    calls = Vector{Union{Nothing, typeof(call)}}(undef, number_of_launches)
    fill!(calls, nothing)
    calls[1] = call
    threads = zeros(Int, number_of_launches)
    blocks = zeros(Int, number_of_launches)
    threads[1] = first_threads
    blocks[1] = first_blocks
    return CUDASmootherKernel(kernel, compiled, calls, threads, blocks)
end

@generated function rebind_smoother_call(call, arguments::A) where {A <: Tuple}
    updates = Expr[]
    for i in 1:fieldcount(A)
        call_index = i + 1
        push!(updates, quote
            if current.source.arguments[$call_index] !== arguments[$i]
                current = CUDA.rebind(current, arguments[$i], $call_index)
            end
        end)
    end
    return quote
        current = call
        $(updates...)
        current
    end
end

function KAPreconditioners.launch_smoother_kernel(
        prepared::CUDASmootherKernel{K, C, KC}, arguments...;
        ndrange, launch_index = 1
    ) where {K, C, KC}
    call = prepared.calls[launch_index]
    if isnothing(call)
        context, threads, blocks = cuda_smoother_launch_config(
            prepared.kernel, ndrange
        )
        call = CUDA.KernelCall(prepared.kernel.f, context, arguments...)
        if call isa KC
            prepared.calls[launch_index] = call
            prepared.threads[launch_index] = threads
            prepared.blocks[launch_index] = blocks
        end
    else
        call = rebind_smoother_call(call, arguments)
        if call isa KC
            prepared.calls[launch_index] = call
        end
        threads = prepared.threads[launch_index]
        blocks = prepared.blocks[launch_index]
    end
    iszero(blocks) && return nothing
    cacheable = call isa KC
    # The first level establishes CUDA ownership for every buffer in this
    # sweep. Remaining levels use the same buffers and stream, so launch their
    # retained converted arguments without repeating the ownership sort.
    if launch_index == 1 || !cacheable
        CUDA.kernel_launch(
            prepared.compiled, call; threads = threads, blocks = blocks
        )
    else
        GC.@preserve call CUDA.kernel_launch(
            call.backend, prepared.compiled, call.arguments;
            threads = threads, blocks = blocks
        )
    end
    return nothing
end

KAPreconditioners.native_dense_lu(::CuArray) = true

KAPreconditioners.replaced_backend_storage_cleanup_threshold(
    ::CUDA.CUDABackend
) = 256 * 1024^2

function KAPreconditioners.release_replaced_backend_storage!(
        ::CUDA.CUDABackend
    )
    # A symbolic AMG rebuild can replace gigabytes of device buffers while the
    # corresponding small CuArray wrappers do not put meaningful pressure on
    # Julia's host GC. The AMG hierarchy accumulates the sizes of allocations
    # it actually replaces and invokes this hook only after that garbage reaches
    # the CUDA cleanup threshold, amortizing collection across small rebuilds.
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

const CUSPARSEValue = Union{Float32, Float64, ComplexF32, ComplexF64}

function KAPreconditioners.build_vendor_ilu(
        A::StaticSparsityMatrixCSR{Tv, Ti, V, I, R, B}
    ) where {Tv, Ti <: Integer, V, I, R, B <: CUDA.CUDABackend}
    scalar_type = KAPreconditioners.matrix_scalar_type(Tv)
    scalar_type <: CUSPARSEValue || throw(
        ArgumentError(
            "CUDA VendorILU does not support matrix scalar type $scalar_type"
        )
    )
    factor_values = copy(A.nzval)
    rowptr = copy(A.rowptr)
    colval = copy(A.colval)
    factor = if Tv <: Number
        CuSparseMatrixCSR{Tv, Ti}(
            rowptr, colval, factor_values, size(A)
        )
    elseif Tv <: StaticMatrix
        block_rows, block_columns = size(Tv)
        block_rows == block_columns || throw(
            DimensionMismatch("VendorILU requires square matrix blocks")
        )
        scalar_values = reinterpret(scalar_type, factor_values)
        dimensions = (
            block_rows * size(A, 1), block_columns * size(A, 2)
        )
        CuSparseMatrixBSR{scalar_type, Ti}(
            rowptr, colval, scalar_values, dimensions,
            block_rows, 'C', length(factor_values)
        )
    else
        throw(
            ArgumentError(
                "CUDA VendorILU requires scalar or static-matrix values, got $Tv"
            )
        )
    end
    KAPreconditioners.refactor_vendor_ilu!(factor)
    return factor, factor_values
end

function KAPreconditioners.refactor_vendor_ilu!(
        factor::Union{CuSparseMatrixCSR, CuSparseMatrixBSR}
    )
    CUDA.CUSPARSE.ilu02!(factor)
    return factor
end

function KAPreconditioners.vendor_ilu_factor_storage_bytes(
        factor::Union{CuSparseMatrixCSR, CuSparseMatrixBSR}
    )
    return sizeof(eltype(factor.rowPtr)) * length(factor.rowPtr) +
        sizeof(eltype(factor.colVal)) * length(factor.colVal)
end

function cusparse_wrapper(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    return CuSparseMatrixCSR{Tv, Ti}(A.rowptr, A.colval, A.nzval, size(A))
end

function KAPreconditioners.csr_matrix(
        A::CuSparseMatrixCSR;
        block_size::Integer = 128
    )
    return StaticSparsityMatrixCSR(
        A.nzVal, A.colVal, A.rowPtr, size(A, 1), size(A, 2),
        KernelAbstractions.get_backend(A.nzVal);
        nthreads = 1, minbatch = Int(block_size), thread_type = :serial
    )
end

function Jutul.KernelExecution.factorize_linear_system(
        ::typeof(lu),
        matrix::StaticSparsityMatrixCSR{
            Tv, Ti, V, I, R, B,
        }
    ) where {Tv, Ti <: Integer, V, I, R, B <: CUDA.CUDABackend}
    return KAPreconditioners.build_coarse_solver(
        matrix, KAPreconditioners.matrix_backend(matrix)
    )
end

function LinearAlgebra.mul!(
        y::CuArray{Tv, 1},
        A::StaticSparsityMatrixCSR{
            Tv, Ti, <:CuArray, <:CuArray, <:CuArray,
        },
        x::CuArray{Tv, 1}
    ) where {Tv <: CUSPARSEValue, Ti}
    length(y) == size(A, 1) || throw(DimensionMismatch())
    length(x) == size(A, 2) || throw(DimensionMismatch())
    return mul!(y, cusparse_wrapper(A), x)
end

function Jutul.maybe_convert_evaluation_state(
        state::Jutul.ImmutableJutulStorage,
        context::Jutul.KernelAbstractionsContext{<:CUDA.CUDABackend}
    )
    converted = Adapt.adapt(KernelAdaptor(), Jutul.data(state))
    return Jutul.ImmutableJutulStorage(converted)
end

# Evaluation states are converted once during simulator transfer. Treat the
# immutable wrapper as an already device-compatible kernel argument thereafter.
function Adapt.adapt_structure(
        ::KernelAdaptor, state::Jutul.ImmutableJutulStorage
    )
    return state
end

function Jutul.KernelExecution.maybe_convert_cross_term_evaluation(
        plan::Jutul.KernelExecution.PreparedCrossTermEvaluation,
        ::Jutul.KernelAbstractionsContext{<:CUDA.CUDABackend}
    )
    converted = Adapt.adapt(KernelAdaptor(), plan)
    if !isbitstype(typeof(converted))
        error("CUDA cross-term evaluation plan must be an isbits type")
    end
    return converted
end

# Jutul's threaded loop only needs a one-dimensional CUDA kernel. KernelCall
# converts its callable and arguments once and retains their host owners through
# the launch.
function jutul_threaded_loop_kernel(f, n::Int)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if index <= n
        @inbounds f(Int(index))
    end
    return nothing
end

function Jutul.KernelExecution.launch_threaded_loop(
        f, n,
        context::Jutul.KernelAbstractionsContext{<:CUDA.CUDABackend};
        cpu_minbatch::Int = Jutul.minbatch(context)
    )
    if n <= 0
        return nothing
    end
    n = Int(n)
    call = CUDA.KernelCall(jutul_threaded_loop_kernel, f, n)
    kernel = CUDA.kernel_compile(
        call;
        always_inline = context.backend.always_inline,
        maxthreads = context.workgroupsize
    )
    threads = min(n, context.workgroupsize)
    blocks = cld(n, threads)
    CUDA.kernel_launch(kernel, call; threads = threads, blocks = blocks)
    return nothing
end

function jutul_preconverted_threaded_loop_kernel(f, n::Int, args...)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if index <= n
        @inbounds f(Int(index), args...)
    end
    return nothing
end

# The prepared cross-term plan already contains device-compatible callables.
# Let KernelCall handle the launch without adapting those callables a second
# time.
struct PreconvertedKernelArgument{F}
    value::F
end

Adapt.adapt_structure(::KernelAdaptor, arg::PreconvertedKernelArgument) =
    arg.value

function Jutul.KernelExecution.launch_preconverted_threaded_loop(
        f, n,
        context::Jutul.KernelAbstractionsContext{<:CUDA.CUDABackend},
        args...
    )
    if n <= 0
        return nothing
    end
    n = Int(n)
    call = CUDA.KernelCall(
        jutul_preconverted_threaded_loop_kernel,
        PreconvertedKernelArgument(f), n, args...
    )
    kernel = CUDA.kernel_compile(
        call;
        always_inline = context.backend.always_inline,
        maxthreads = context.workgroupsize
    )
    threads = min(n, context.workgroupsize)
    blocks = cld(n, threads)
    CUDA.kernel_launch(kernel, call; threads = threads, blocks = blocks)
    return nothing
end

end
