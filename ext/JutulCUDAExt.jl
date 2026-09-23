module JutulCUDAExt

using Jutul
using Jutul.KAPreconditioners
using Jutul: StaticSparsityMatrixCSR
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixBSR, CuSparseMatrixCSR
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using StaticArrays: StaticMatrix
import Adapt
import CUDA: KernelAdaptor
import KernelAbstractions as KA

include("cuda_sparse_lu.jl")

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
        push!(
            updates, quote
                if current.source.arguments[$call_index] !== arguments[$i]
                    current = CUDA.rebind(current, arguments[$i], $call_index)
                end
            end
        )
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

mutable struct CUDABSRSolveInfo
    info::CUDA.CUSPARSE.bsrsv2Info_t
    function CUDABSRSolveInfo()
        info = Ref{CUDA.CUSPARSE.bsrsv2Info_t}()
        CUDA.CUSPARSE.cusparseCreateBsrsv2Info(info)
        object = new(info[])
        finalizer(object) do current
            CUDA.CUSPARSE.cusparseDestroyBsrsv2Info(current.info)
        end
        return object
    end
end

Base.unsafe_convert(
    ::Type{CUDA.CUSPARSE.bsrsv2Info_t}, info::CUDABSRSolveInfo
) = info.info

struct CUDACSRVendorILUFactor{M, FI, FD, MD, DD, SI, W}
    matrix::M
    factor_info::FI
    factor_descriptor::FD
    lower_matrix_descriptor::MD
    upper_matrix_descriptor::MD
    input_descriptor::DD
    output_descriptor::DD
    lower_solve_info::SI
    upper_solve_info::SI
    factor_workspace::W
    lower_workspace::W
    upper_workspace::W
end

struct CUDABSRVendorILUFactor{M, FI, FD, SI, SD, W}
    matrix::M
    factor_info::FI
    factor_descriptor::FD
    lower_solve_info::SI
    lower_solve_descriptor::SD
    upper_solve_info::SI
    upper_solve_descriptor::SD
    factor_workspace::W
    lower_workspace::W
    upper_workspace::W
end

const CUDAVendorILUFactor = Union{
    CUDACSRVendorILUFactor, CUDABSRVendorILUFactor,
}

cuda_csr_ilu_functions(::Type{Float32}) = (
    CUDA.CUSPARSE.cusparseScsrilu02_bufferSize,
    CUDA.CUSPARSE.cusparseScsrilu02_analysis,
    CUDA.CUSPARSE.cusparseScsrilu02,
)
cuda_csr_ilu_functions(::Type{Float64}) = (
    CUDA.CUSPARSE.cusparseDcsrilu02_bufferSize,
    CUDA.CUSPARSE.cusparseDcsrilu02_analysis,
    CUDA.CUSPARSE.cusparseDcsrilu02,
)
cuda_csr_ilu_functions(::Type{ComplexF32}) = (
    CUDA.CUSPARSE.cusparseCcsrilu02_bufferSize,
    CUDA.CUSPARSE.cusparseCcsrilu02_analysis,
    CUDA.CUSPARSE.cusparseCcsrilu02,
)
cuda_csr_ilu_functions(::Type{ComplexF64}) = (
    CUDA.CUSPARSE.cusparseZcsrilu02_bufferSize,
    CUDA.CUSPARSE.cusparseZcsrilu02_analysis,
    CUDA.CUSPARSE.cusparseZcsrilu02,
)

cuda_bsr_ilu_functions(::Type{Float32}) = (
    CUDA.CUSPARSE.cusparseSbsrilu02_bufferSize,
    CUDA.CUSPARSE.cusparseSbsrilu02_analysis,
    CUDA.CUSPARSE.cusparseSbsrilu02,
)
cuda_bsr_ilu_functions(::Type{Float64}) = (
    CUDA.CUSPARSE.cusparseDbsrilu02_bufferSize,
    CUDA.CUSPARSE.cusparseDbsrilu02_analysis,
    CUDA.CUSPARSE.cusparseDbsrilu02,
)
cuda_bsr_ilu_functions(::Type{ComplexF32}) = (
    CUDA.CUSPARSE.cusparseCbsrilu02_bufferSize,
    CUDA.CUSPARSE.cusparseCbsrilu02_analysis,
    CUDA.CUSPARSE.cusparseCbsrilu02,
)
cuda_bsr_ilu_functions(::Type{ComplexF64}) = (
    CUDA.CUSPARSE.cusparseZbsrilu02_bufferSize,
    CUDA.CUSPARSE.cusparseZbsrilu02_analysis,
    CUDA.CUSPARSE.cusparseZbsrilu02,
)

cuda_bsr_solve_functions(::Type{Float32}) = (
    CUDA.CUSPARSE.cusparseSbsrsv2_bufferSize,
    CUDA.CUSPARSE.cusparseSbsrsv2_analysis,
    CUDA.CUSPARSE.cusparseSbsrsv2_solve,
)
cuda_bsr_solve_functions(::Type{Float64}) = (
    CUDA.CUSPARSE.cusparseDbsrsv2_bufferSize,
    CUDA.CUSPARSE.cusparseDbsrsv2_analysis,
    CUDA.CUSPARSE.cusparseDbsrsv2_solve,
)
cuda_bsr_solve_functions(::Type{ComplexF32}) = (
    CUDA.CUSPARSE.cusparseCbsrsv2_bufferSize,
    CUDA.CUSPARSE.cusparseCbsrsv2_analysis,
    CUDA.CUSPARSE.cusparseCbsrsv2_solve,
)
cuda_bsr_solve_functions(::Type{ComplexF64}) = (
    CUDA.CUSPARSE.cusparseZbsrsv2_bufferSize,
    CUDA.CUSPARSE.cusparseZbsrsv2_analysis,
    CUDA.CUSPARSE.cusparseZbsrsv2_solve,
)

function cuda_csr_ilu_buffer_size(matrix, info, descriptor)
    buffer_size, _, _ = cuda_csr_ilu_functions(eltype(matrix.nzVal))
    output = Ref{Cint}()
    buffer_size(
        CUDA.CUSPARSE.handle(), size(matrix, 1), length(matrix.nzVal),
        descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal, info, output
    )
    return Int(output[])
end

function cuda_bsr_ilu_buffer_size(matrix, info, descriptor)
    buffer_size, _, _ = cuda_bsr_ilu_functions(eltype(matrix.nzVal))
    output = Ref{Cint}()
    block_rows = div(size(matrix, 1), matrix.blockDim)
    buffer_size(
        CUDA.CUSPARSE.handle(), matrix.dir, block_rows, matrix.nnzb,
        descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        matrix.blockDim, info, output
    )
    return Int(output[])
end

function cuda_bsr_solve_buffer_size(matrix, info, descriptor)
    buffer_size, _, _ = cuda_bsr_solve_functions(eltype(matrix.nzVal))
    output = Ref{Cint}()
    block_rows = div(size(matrix, 1), matrix.blockDim)
    buffer_size(
        CUDA.CUSPARSE.handle(), matrix.dir, 'N', block_rows, matrix.nnzb,
        descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        matrix.blockDim, info, output
    )
    return Int(output[])
end

function cuda_set_triangular_attributes!(descriptor, uplo::Char, diag::Char)
    fill_mode = Ref{CUDA.CUSPARSE.cusparseFillMode_t}(uplo)
    diagonal_type = Ref{CUDA.CUSPARSE.cusparseDiagType_t}(diag)
    CUDA.CUSPARSE.cusparseSpMatSetAttribute(
        descriptor, 'F', fill_mode, Csize_t(sizeof(fill_mode))
    )
    CUDA.CUSPARSE.cusparseSpMatSetAttribute(
        descriptor, 'D', diagonal_type, Csize_t(sizeof(diagonal_type))
    )
    return descriptor
end

function cuda_csr_solve_buffer_size(
        matrix_descriptor, input_descriptor, output_descriptor, solve_info,
        ::Type{T}
    ) where {T}
    output = Ref{Csize_t}()
    CUDA.CUSPARSE.cusparseSpSV_bufferSize(
        CUDA.CUSPARSE.handle(), 'N', Ref{T}(one(T)), matrix_descriptor,
        input_descriptor, output_descriptor, T,
        CUDA.CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT, solve_info, output
    )
    return Int(output[])
end

function cuda_check_ilu_pivot(info, block::Bool)
    position = Ref{Cint}(1)
    if block
        CUDA.CUSPARSE.cusparseXbsrilu02_zeroPivot(
            CUDA.CUSPARSE.handle(), info, position
        )
    else
        CUDA.CUSPARSE.cusparseXcsrilu02_zeroPivot(
            CUDA.CUSPARSE.handle(), info, position
        )
    end
    position[] < 0 || error(
        "Structural zero in vendor ILU factor at row $(position[])"
    )
    return nothing
end

function cuda_check_bsr_solve_pivot(info)
    position = Ref{Cint}(1)
    CUDA.CUSPARSE.cusparseXbsrsv2_zeroPivot(
        CUDA.CUSPARSE.handle(), info, position
    )
    position[] < 0 || error(
        "Structural/numerical zero in vendor triangular solve at row " *
            "$(position[])"
    )
    return nothing
end

function cuda_analyze_vendor_ilu(matrix::CuSparseMatrixCSR, solve_values)
    T = eltype(matrix.nzVal)
    factor_info = CUDA.CUSPARSE.ILU0Info()
    factor_descriptor = CUDA.CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', 'O')
    lower_matrix_descriptor = cuda_set_triangular_attributes!(
        CUDA.CUSPARSE.CuSparseMatrixDescriptor(matrix, 'O'), 'L', 'U'
    )
    upper_matrix_descriptor = cuda_set_triangular_attributes!(
        CUDA.CUSPARSE.CuSparseMatrixDescriptor(matrix, 'O'), 'U', 'N'
    )
    input_descriptor = CUDA.CUSPARSE.CuDenseVectorDescriptor(solve_values)
    output_descriptor = CUDA.CUSPARSE.CuDenseVectorDescriptor(solve_values)
    lower_solve_info = CUDA.CUSPARSE.CuSparseSpSVDescriptor()
    upper_solve_info = CUDA.CUSPARSE.CuSparseSpSVDescriptor()
    factor_size = cuda_csr_ilu_buffer_size(
        matrix, factor_info, factor_descriptor
    )
    lower_size = cuda_csr_solve_buffer_size(
        lower_matrix_descriptor, input_descriptor, output_descriptor,
        lower_solve_info, T
    )
    upper_size = cuda_csr_solve_buffer_size(
        upper_matrix_descriptor, input_descriptor, output_descriptor,
        upper_solve_info, T
    )
    factor_workspace = CuVector{UInt8}(undef, factor_size)
    lower_workspace = CuVector{UInt8}(undef, lower_size)
    upper_workspace = CuVector{UInt8}(undef, upper_size)
    factor = CUDACSRVendorILUFactor(
        matrix, factor_info, factor_descriptor,
        lower_matrix_descriptor, upper_matrix_descriptor,
        input_descriptor, output_descriptor,
        lower_solve_info, upper_solve_info,
        factor_workspace, lower_workspace, upper_workspace
    )

    _, analyze, _ = cuda_csr_ilu_functions(T)
    analyze(
        CUDA.CUSPARSE.handle(), size(matrix, 1), length(matrix.nzVal),
        factor_descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        factor_info, CUDA.CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL,
        factor_workspace
    )
    cuda_check_ilu_pivot(factor_info, false)

    for (matrix_descriptor, solve_info, solve_workspace) in (
            (lower_matrix_descriptor, lower_solve_info, lower_workspace),
            (upper_matrix_descriptor, upper_solve_info, upper_workspace),
        )
        CUDA.CUSPARSE.cusparseSpSV_analysis(
            CUDA.CUSPARSE.handle(), 'N', Ref{T}(one(T)), matrix_descriptor,
            input_descriptor, output_descriptor, T,
            CUDA.CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT, solve_info,
            solve_workspace
        )
    end
    KAPreconditioners.refactor_vendor_ilu!(factor)
    return factor
end

function cuda_analyze_vendor_ilu(matrix::CuSparseMatrixBSR, solve_values)
    T = eltype(matrix.nzVal)
    factor_info = CUDA.CUSPARSE.ILU0InfoBSR()
    factor_descriptor = CUDA.CUSPARSE.CuMatrixDescriptor('G', 'U', 'N', 'O')
    lower_solve_info = CUDABSRSolveInfo()
    lower_solve_descriptor = CUDA.CUSPARSE.CuMatrixDescriptor(
        'G', 'L', 'U', 'O'
    )
    upper_solve_info = CUDABSRSolveInfo()
    upper_solve_descriptor = CUDA.CUSPARSE.CuMatrixDescriptor(
        'G', 'U', 'N', 'O'
    )
    factor_size = cuda_bsr_ilu_buffer_size(
        matrix, factor_info, factor_descriptor
    )
    lower_size = cuda_bsr_solve_buffer_size(
        matrix, lower_solve_info, lower_solve_descriptor
    )
    upper_size = cuda_bsr_solve_buffer_size(
        matrix, upper_solve_info, upper_solve_descriptor
    )
    factor_workspace = CuVector{UInt8}(undef, factor_size)
    lower_workspace = CuVector{UInt8}(undef, lower_size)
    upper_workspace = CuVector{UInt8}(undef, upper_size)
    factor = CUDABSRVendorILUFactor(
        matrix, factor_info, factor_descriptor,
        lower_solve_info, lower_solve_descriptor,
        upper_solve_info, upper_solve_descriptor,
        factor_workspace, lower_workspace, upper_workspace
    )

    _, analyze_factor, _ = cuda_bsr_ilu_functions(T)
    block_rows = div(size(matrix, 1), matrix.blockDim)
    analyze_factor(
        CUDA.CUSPARSE.handle(), matrix.dir, block_rows, matrix.nnzb,
        factor_descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        matrix.blockDim, factor_info,
        CUDA.CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, factor_workspace
    )
    cuda_check_ilu_pivot(factor_info, true)
    KAPreconditioners.refactor_vendor_ilu!(factor)

    _, analyze_solve, _ = cuda_bsr_solve_functions(T)
    for (descriptor, info, solve_workspace) in (
            (lower_solve_descriptor, lower_solve_info, lower_workspace),
            (upper_solve_descriptor, upper_solve_info, upper_workspace),
        )
        analyze_solve(
            CUDA.CUSPARSE.handle(), matrix.dir, 'N', block_rows, matrix.nnzb,
            descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
            matrix.blockDim, info,
            CUDA.CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, solve_workspace
        )
        cuda_check_bsr_solve_pivot(info)
    end
    return factor
end

function KAPreconditioners.build_vendor_ilu(
        A::StaticSparsityMatrixCSR{Tv, Ti, V, I, R, B}, work
    ) where {Tv, Ti <: Integer, V, I, R, B <: CUDA.CUDABackend}
    scalar_type = KAPreconditioners.matrix_scalar_type(Tv)
    scalar_type <: CUSPARSEValue || throw(
        ArgumentError(
            "CUDA VendorILU does not support matrix scalar type $scalar_type"
        )
    )
    maximum_index = max(size(A)..., length(A.nzval) + 1)
    maximum_index <= typemax(Cint) || throw(
        ArgumentError(
            "CUDA VendorILU requires a matrix indexable by 32-bit integers"
        )
    )
    factor_values = copy(A.nzval)
    # CUDA.jl's ilu02! and BSR triangular solves use the legacy cuSPARSE
    # routines. Those routines accept 32-bit indices only, even though the
    # sparse wrapper itself permits other integer types. Convert the private
    # factor pattern to Cint and use the one-parameter constructors; passing
    # Int64 buffers to the legacy C API makes it reinterpret them as packed
    # Int32 indices and can result in an illegal memory access.
    rowptr = convert(CuVector{Cint}, A.rowptr)
    colval = convert(CuVector{Cint}, A.colval)
    factor = if Tv <: Number
        CuSparseMatrixCSR{Tv}(
            rowptr, colval, factor_values, size(A)
        )
    elseif Tv <: StaticMatrix
        block_rows, block_columns = size(Tv)
        block_rows == block_columns || throw(
            DimensionMismatch("VendorILU requires square matrix blocks")
        )
        scalar_values = reinterpret(scalar_type, factor_values)
        dimensions = (
            block_rows * size(A, 1), block_columns * size(A, 2),
        )
        CuSparseMatrixBSR{scalar_type}(
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
    solve_values = KAPreconditioners.vendor_ilu_vector(work)
    analyzed_factor = cuda_analyze_vendor_ilu(factor, solve_values)
    return analyzed_factor, factor_values
end

function KAPreconditioners.refactor_vendor_ilu!(
        factor::CUDACSRVendorILUFactor
    )
    matrix = factor.matrix
    _, _, refactor = cuda_csr_ilu_functions(eltype(matrix.nzVal))
    refactor(
        CUDA.CUSPARSE.handle(), size(matrix, 1), length(matrix.nzVal),
        factor.factor_descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        factor.factor_info, CUDA.CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL,
        factor.factor_workspace
    )
    for solve_info in (factor.lower_solve_info, factor.upper_solve_info)
        CUDA.CUSPARSE.cusparseSpSV_updateMatrix(
            CUDA.CUSPARSE.handle(), solve_info, matrix.nzVal,
            CUDA.CUSPARSE.CUSPARSE_SPSV_UPDATE_GENERAL
        )
    end
    return factor
end

function KAPreconditioners.refactor_vendor_ilu!(
        factor::CUDABSRVendorILUFactor
    )
    matrix = factor.matrix
    _, _, refactor = cuda_bsr_ilu_functions(eltype(matrix.nzVal))
    block_rows = div(size(matrix, 1), matrix.blockDim)
    refactor(
        CUDA.CUSPARSE.handle(), matrix.dir, block_rows, matrix.nnzb,
        factor.factor_descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        matrix.blockDim, factor.factor_info,
        CUDA.CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL,
        factor.factor_workspace
    )
    return factor
end

function KAPreconditioners.vendor_ilu_factor_storage_bytes(
        factor::CUDAVendorILUFactor
    )
    matrix = factor.matrix
    return sizeof(eltype(matrix.rowPtr)) * length(matrix.rowPtr) +
        sizeof(eltype(matrix.colVal)) * length(matrix.colVal) +
        sum(
        sizeof(eltype(workspace)) * length(workspace)
            for workspace in (
                factor.factor_workspace,
                factor.lower_workspace,
                factor.upper_workspace,
            )
    )
end

function KAPreconditioners.solve_vendor_ilu_factor!(
        values, factor::CUDACSRVendorILUFactor
    )
    T = eltype(values)
    CUDA.CUSPARSE.cusparseDnVecSetValues(factor.input_descriptor, values)
    CUDA.CUSPARSE.cusparseDnVecSetValues(factor.output_descriptor, values)
    for (matrix_descriptor, solve_info) in (
            (factor.lower_matrix_descriptor, factor.lower_solve_info),
            (factor.upper_matrix_descriptor, factor.upper_solve_info),
        )
        CUDA.CUSPARSE.cusparseSpSV_solve(
            CUDA.CUSPARSE.handle(), 'N', Ref{T}(one(T)), matrix_descriptor,
            factor.input_descriptor, factor.output_descriptor, T,
            CUDA.CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT, solve_info
        )
    end
    return values
end

function KAPreconditioners.solve_vendor_ilu_factor!(
        values, factor::CUDABSRVendorILUFactor
    )
    matrix = factor.matrix
    T = eltype(values)
    _, _, solve = cuda_bsr_solve_functions(T)
    block_rows = div(size(matrix, 1), matrix.blockDim)
    for (descriptor, info, solve_workspace) in (
            (
                factor.lower_solve_descriptor, factor.lower_solve_info,
                factor.lower_workspace,
            ),
            (
                factor.upper_solve_descriptor, factor.upper_solve_info,
                factor.upper_workspace,
            ),
        )
        solve(
            CUDA.CUSPARSE.handle(), matrix.dir, 'N', block_rows, matrix.nnzb,
            Ref{T}(one(T)), descriptor, matrix.nzVal,
            matrix.rowPtr, matrix.colVal, matrix.blockDim, info,
            values, values, CUDA.CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            solve_workspace
        )
    end
    return values
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
