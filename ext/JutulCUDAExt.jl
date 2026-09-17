module JutulCUDAExt

using Jutul
using Jutul.KAPreconditioners
using Jutul: StaticSparsityMatrixCSR
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using KernelAbstractions
using LinearAlgebra
import Adapt
import CUDA: KernelAdaptor

KAPreconditioners.native_dense_lu(::CuArray) = true

const CUSPARSEValue = Union{Float32, Float64, ComplexF32, ComplexF64}

function cusparse_wrapper(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    return CuSparseMatrixCSR{Tv, Ti}(A.rowptr, A.colval, A.nzval, size(A))
end

function KAPreconditioners.csr_matrix(A::CuSparseMatrixCSR;
        block_size::Integer = 128)
    return StaticSparsityMatrixCSR(
        A.nzVal, A.colVal, A.rowPtr, size(A, 1), size(A, 2),
        KernelAbstractions.get_backend(A.nzVal);
        nthreads = 1, minbatch = Int(block_size), thread_type = :serial)
end

function LinearAlgebra.mul!(y::CuArray{Tv, 1},
        A::StaticSparsityMatrixCSR{
            Tv, Ti, <:CuArray, <:CuArray, <:CuArray},
        x::CuArray{Tv, 1}) where {Tv<:CUSPARSEValue, Ti}
    length(y) == size(A, 1) || throw(DimensionMismatch())
    length(x) == size(A, 2) || throw(DimensionMismatch())
    return mul!(y, cusparse_wrapper(A), x)
end

function Jutul.maybe_convert_evaluation_state(
        state::Jutul.ImmutableJutulStorage,
        context::Jutul.KernelAbstractionsContext{<:CUDA.CUDABackend})
    converted = Adapt.adapt(KernelAdaptor(), Jutul.data(state))
    return Jutul.ImmutableJutulStorage(converted)
end

# Evaluation states are converted once during simulator transfer. Treat the
# immutable wrapper as an already device-compatible kernel argument thereafter.
function Adapt.adapt_structure(
        ::KernelAdaptor, state::Jutul.ImmutableJutulStorage)
    return state
end

function Jutul.maybe_convert_cross_term_evaluation(
        plan::Jutul.PreparedCrossTermEvaluation,
        ::Jutul.KernelAbstractionsContext{<:CUDA.CUDABackend})
    converted = Adapt.adapt(KernelAdaptor(), plan)
    if !isbitstype(typeof(converted))
        error("CUDA cross-term evaluation plan must be an isbits type")
    end
    return converted
end

# KernelAbstractions' CUDA launcher converts arguments when constructing the
# kernel and again when launching it. Jutul's threaded loop only needs a
# one-dimensional CUDA kernel, so convert its callable once and launch the
# compiled kernel with argument conversion disabled.
function jutul_threaded_loop_kernel(f, n::Int)
    index = (CUDA.blockIdx().x - 1)*CUDA.blockDim().x + CUDA.threadIdx().x
    if index <= n
        @inbounds f(Int(index))
    end
    return nothing
end

function Jutul.KernelExecution.launch_threaded_loop(f, n,
        context::Jutul.KernelAbstractionsContext{<:CUDA.CUDABackend};
        cpu_minbatch::Int = Jutul.minbatch(context))
    if n <= 0
        return nothing
    end
    n = Int(n)
    device_f = CUDA.cudaconvert(f)
    kernel = CUDA.cufunction(
        jutul_threaded_loop_kernel, Tuple{typeof(device_f), Int};
        always_inline = context.backend.always_inline,
        maxthreads = context.workgroupsize)
    threads = min(n, context.workgroupsize)
    blocks = cld(n, threads)
    GC.@preserve f begin
        kernel(device_f, n;
            threads = threads, blocks = blocks, convert = Val(false))
    end
    return nothing
end

function jutul_preconverted_threaded_loop_kernel(f, n::Int, args...)
    index = (CUDA.blockIdx().x - 1)*CUDA.blockDim().x + CUDA.threadIdx().x
    if index <= n
        @inbounds f(Int(index), args...)
    end
    return nothing
end

function Jutul.launch_preconverted_threaded_loop(f, n,
        context::Jutul.KernelAbstractionsContext{<:CUDA.CUDABackend},
        args...)
    if n <= 0
        return nothing
    end
    n = Int(n)
    argument_types = Tuple{typeof(f), Int, map(typeof, args)...}
    kernel = CUDA.cufunction(
        jutul_preconverted_threaded_loop_kernel, argument_types;
        always_inline = context.backend.always_inline,
        maxthreads = context.workgroupsize)
    threads = min(n, context.workgroupsize)
    blocks = cld(n, threads)
    GC.@preserve f args begin
        kernel(f, n, args...;
            threads = threads, blocks = blocks, convert = Val(false))
    end
    return nothing
end

end
