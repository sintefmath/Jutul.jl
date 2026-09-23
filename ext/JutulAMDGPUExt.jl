module JutulAMDGPUExt

using Jutul
using Jutul.KAPreconditioners
using Jutul: StaticSparsityMatrixCSR
using KernelAbstractions
using AMDGPU
using AMDGPU.rocSPARSE
using LinearAlgebra
using StaticArrays: StaticMatrix

KAPreconditioners.native_dense_lu(::ROCArray) = true

const ROCSPARSEValue = Union{Float32, Float64, ComplexF32, ComplexF64}

struct ROCVendorILUFactor{M, I, D, W}
    matrix::M
    factor_info::I
    factor_descriptor::D
    lower_solve_info::I
    lower_solve_descriptor::D
    upper_solve_info::I
    upper_solve_descriptor::D
    factor_workspace::W
    lower_workspace::W
    upper_workspace::W
end

roc_value_prefix(::Type{Float32}) = "s"
roc_value_prefix(::Type{Float64}) = "d"
roc_value_prefix(::Type{ComplexF32}) = "c"
roc_value_prefix(::Type{ComplexF64}) = "z"

function roc_sparse_function(::Type{T}, operation::Symbol) where {T}
    name = Symbol("rocsparse_", roc_value_prefix(T), operation)
    return getproperty(rocSPARSE, name)
end

function roc_ilu_buffer_size(matrix::ROCSparseMatrixCSR, info, descriptor)
    buffer_size = roc_sparse_function(eltype(matrix.nzVal), :csrilu0_buffer_size)
    output = Ref{Csize_t}()
    buffer_size(
        rocSPARSE.handle(), size(matrix, 1), length(matrix.nzVal),
        descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal, info, output
    )
    return Int(output[])
end

function roc_ilu_buffer_size(matrix::ROCSparseMatrixBSR, info, descriptor)
    buffer_size = roc_sparse_function(eltype(matrix.nzVal), :bsrilu0_buffer_size)
    output = Ref{Csize_t}()
    block_rows = div(size(matrix, 1), matrix.blockDim)
    buffer_size(
        rocSPARSE.handle(), matrix.dir, block_rows, matrix.nnzb,
        descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        matrix.blockDim, info, output
    )
    return Int(output[])
end

function roc_solve_buffer_size(
        matrix::ROCSparseMatrixCSR, info, descriptor
    )
    buffer_size = roc_sparse_function(eltype(matrix.nzVal), :csrsv_buffer_size)
    output = Ref{Csize_t}()
    buffer_size(
        rocSPARSE.handle(), 'N', size(matrix, 1), length(matrix.nzVal),
        descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal, info, output
    )
    return Int(output[])
end

function roc_solve_buffer_size(
        matrix::ROCSparseMatrixBSR, info, descriptor
    )
    buffer_size = roc_sparse_function(eltype(matrix.nzVal), :bsrsv_buffer_size)
    output = Ref{Csize_t}()
    block_rows = div(size(matrix, 1), matrix.blockDim)
    buffer_size(
        rocSPARSE.handle(), matrix.dir, 'N', block_rows, matrix.nnzb,
        descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        matrix.blockDim, info, output
    )
    return Int(output[])
end

function roc_check_ilu_pivot(info, block::Bool)
    position = Ref{Cint}(1)
    pivot = block ? :rocsparse_bsrilu0_zero_pivot :
        :rocsparse_csrilu0_zero_pivot
    getproperty(rocSPARSE, pivot)(rocSPARSE.handle(), info, position)
    position[] < 0 || error(
        "Structural zero in vendor ILU factor at row $(position[])"
    )
    return nothing
end

function roc_check_solve_pivot(info, descriptor, block::Bool)
    position = Ref{Cint}(1)
    if block
        rocSPARSE.rocsparse_bsrsv_zero_pivot(
            rocSPARSE.handle(), info, position
        )
    else
        rocSPARSE.rocsparse_csrsv_zero_pivot(
            rocSPARSE.handle(), descriptor, info, position
        )
    end
    position[] < 0 || error(
        "Structural/numerical zero in vendor triangular solve at row " *
            "$(position[])"
    )
    return nothing
end

function roc_analyze_vendor_ilu(matrix::ROCSparseMatrixCSR)
    factor_info = rocSPARSE.MatInfo()
    factor_descriptor = rocSPARSE.ROCMatrixDescriptor('G', 'L', 'N', 'O')
    lower_solve_info = rocSPARSE.MatInfo()
    lower_solve_descriptor = rocSPARSE.ROCMatrixDescriptor(
        'G', 'L', 'U', 'O'
    )
    upper_solve_info = rocSPARSE.MatInfo()
    upper_solve_descriptor = rocSPARSE.ROCMatrixDescriptor(
        'G', 'U', 'N', 'O'
    )
    factor_workspace = ROCVector{UInt8}(
        undef, roc_ilu_buffer_size(
            matrix, factor_info, factor_descriptor
        )
    )
    lower_workspace = ROCVector{UInt8}(
        undef, roc_solve_buffer_size(
            matrix, lower_solve_info, lower_solve_descriptor
        )
    )
    upper_workspace = ROCVector{UInt8}(
        undef, roc_solve_buffer_size(
            matrix, upper_solve_info, upper_solve_descriptor
        )
    )
    factor = ROCVendorILUFactor(
        matrix, factor_info, factor_descriptor,
        lower_solve_info, lower_solve_descriptor,
        upper_solve_info, upper_solve_descriptor,
        factor_workspace, lower_workspace, upper_workspace
    )

    analyze_factor = roc_sparse_function(
        eltype(matrix.nzVal), :csrilu0_analysis
    )
    analyze_factor(
        rocSPARSE.handle(), size(matrix, 1), length(matrix.nzVal),
        factor_descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        factor_info, rocSPARSE.rocsparse_analysis_policy_force,
        rocSPARSE.rocsparse_solve_policy_auto, factor_workspace
    )
    roc_check_ilu_pivot(factor_info, false)
    KAPreconditioners.refactor_vendor_ilu!(factor)

    analyze_solve = roc_sparse_function(
        eltype(matrix.nzVal), :csrsv_analysis
    )
    for (descriptor, info, workspace) in (
            (lower_solve_descriptor, lower_solve_info, lower_workspace),
            (upper_solve_descriptor, upper_solve_info, upper_workspace),
        )
        analyze_solve(
            rocSPARSE.handle(), 'N', size(matrix, 1), length(matrix.nzVal),
            descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal, info,
            rocSPARSE.rocsparse_analysis_policy_force,
            rocSPARSE.rocsparse_solve_policy_auto, workspace
        )
        roc_check_solve_pivot(info, descriptor, false)
    end
    return factor
end

function roc_analyze_vendor_ilu(matrix::ROCSparseMatrixBSR)
    factor_info = rocSPARSE.MatInfo()
    factor_descriptor = rocSPARSE.ROCMatrixDescriptor('G', 'U', 'N', 'O')
    lower_solve_info = rocSPARSE.MatInfo()
    lower_solve_descriptor = rocSPARSE.ROCMatrixDescriptor(
        'G', 'L', 'U', 'O'
    )
    upper_solve_info = rocSPARSE.MatInfo()
    upper_solve_descriptor = rocSPARSE.ROCMatrixDescriptor(
        'G', 'U', 'N', 'O'
    )
    factor_workspace = ROCVector{UInt8}(
        undef, roc_ilu_buffer_size(
            matrix, factor_info, factor_descriptor
        )
    )
    lower_workspace = ROCVector{UInt8}(
        undef, roc_solve_buffer_size(
            matrix, lower_solve_info, lower_solve_descriptor
        )
    )
    upper_workspace = ROCVector{UInt8}(
        undef, roc_solve_buffer_size(
            matrix, upper_solve_info, upper_solve_descriptor
        )
    )
    factor = ROCVendorILUFactor(
        matrix, factor_info, factor_descriptor,
        lower_solve_info, lower_solve_descriptor,
        upper_solve_info, upper_solve_descriptor,
        factor_workspace, lower_workspace, upper_workspace
    )

    block_rows = div(size(matrix, 1), matrix.blockDim)
    analyze_factor = roc_sparse_function(
        eltype(matrix.nzVal), :bsrilu0_analysis
    )
    analyze_factor(
        rocSPARSE.handle(), matrix.dir, block_rows, matrix.nnzb,
        factor_descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        matrix.blockDim, factor_info,
        rocSPARSE.rocsparse_analysis_policy_force,
        rocSPARSE.rocsparse_solve_policy_auto, factor_workspace
    )
    roc_check_ilu_pivot(factor_info, true)
    KAPreconditioners.refactor_vendor_ilu!(factor)

    analyze_solve = roc_sparse_function(
        eltype(matrix.nzVal), :bsrsv_analysis
    )
    for (descriptor, info, workspace) in (
            (lower_solve_descriptor, lower_solve_info, lower_workspace),
            (upper_solve_descriptor, upper_solve_info, upper_workspace),
        )
        analyze_solve(
            rocSPARSE.handle(), matrix.dir, 'N', block_rows, matrix.nnzb,
            descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
            matrix.blockDim, info,
            rocSPARSE.rocsparse_analysis_policy_force,
            rocSPARSE.rocsparse_solve_policy_auto, workspace
        )
        roc_check_solve_pivot(info, descriptor, true)
    end
    return factor
end

function KAPreconditioners.build_vendor_ilu(
        A::StaticSparsityMatrixCSR{Tv, Ti, V, I, R, B}, _work
    ) where {Tv, Ti <: Integer, V, I, R, B <: AMDGPU.ROCBackend}
    scalar_type = KAPreconditioners.matrix_scalar_type(Tv)
    scalar_type <: ROCSPARSEValue || throw(
        ArgumentError(
            "AMDGPU VendorILU does not support matrix scalar type $scalar_type"
        )
    )
    maximum_index = max(size(A)..., length(A.nzval) + 1)
    maximum_index <= typemax(Cint) || throw(
        ArgumentError(
            "AMDGPU VendorILU requires a matrix indexable by 32-bit integers"
        )
    )
    factor_values = copy(A.nzval)
    rowptr = convert(ROCVector{Cint}, A.rowptr)
    colval = convert(ROCVector{Cint}, A.colval)
    factor = if Tv <: Number
        ROCSparseMatrixCSR{Tv}(
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
        ROCSparseMatrixBSR{scalar_type}(
            rowptr, colval, scalar_values, dimensions,
            block_rows, 'C', length(factor_values)
        )
    else
        throw(
            ArgumentError(
                "AMDGPU VendorILU requires scalar or static-matrix values, got $Tv"
            )
        )
    end
    analyzed_factor = roc_analyze_vendor_ilu(factor)
    return analyzed_factor, factor_values
end

function KAPreconditioners.refactor_vendor_ilu!(
        factor::ROCVendorILUFactor{<:ROCSparseMatrixCSR}
    )
    matrix = factor.matrix
    refactor = roc_sparse_function(eltype(matrix.nzVal), :csrilu0)
    refactor(
        rocSPARSE.handle(), size(matrix, 1), length(matrix.nzVal),
        factor.factor_descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        factor.factor_info, rocSPARSE.rocsparse_solve_policy_auto,
        factor.factor_workspace
    )
    return factor
end

function KAPreconditioners.refactor_vendor_ilu!(
        factor::ROCVendorILUFactor{<:ROCSparseMatrixBSR}
    )
    matrix = factor.matrix
    block_rows = div(size(matrix, 1), matrix.blockDim)
    refactor = roc_sparse_function(eltype(matrix.nzVal), :bsrilu0)
    refactor(
        rocSPARSE.handle(), matrix.dir, block_rows, matrix.nnzb,
        factor.factor_descriptor, matrix.nzVal, matrix.rowPtr, matrix.colVal,
        matrix.blockDim, factor.factor_info,
        rocSPARSE.rocsparse_solve_policy_auto, factor.factor_workspace
    )
    return factor
end

function KAPreconditioners.vendor_ilu_factor_storage_bytes(
        factor::ROCVendorILUFactor
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
        values, factor::ROCVendorILUFactor{<:ROCSparseMatrixCSR}
    )
    matrix = factor.matrix
    T = eltype(values)
    solve = roc_sparse_function(T, :csrsv_solve)
    for (descriptor, info, workspace) in (
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
            rocSPARSE.handle(), 'N', size(matrix, 1), length(matrix.nzVal),
            Ref{T}(one(T)), descriptor, matrix.nzVal,
            matrix.rowPtr, matrix.colVal, info, values, values,
            rocSPARSE.rocsparse_solve_policy_auto, workspace
        )
    end
    return values
end

function KAPreconditioners.solve_vendor_ilu_factor!(
        values, factor::ROCVendorILUFactor{<:ROCSparseMatrixBSR}
    )
    matrix = factor.matrix
    T = eltype(values)
    block_rows = div(size(matrix, 1), matrix.blockDim)
    solve = roc_sparse_function(T, :bsrsv_solve)
    for (descriptor, info, workspace) in (
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
            rocSPARSE.handle(), matrix.dir, 'N', block_rows, matrix.nnzb,
            Ref{T}(one(T)), descriptor, matrix.nzVal,
            matrix.rowPtr, matrix.colVal, matrix.blockDim, info,
            values, values, rocSPARSE.rocsparse_solve_policy_auto, workspace
        )
    end
    return values
end

function KAPreconditioners.csr_matrix(
        A::ROCSparseMatrixCSR;
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
    ) where {Tv, Ti <: Integer, V, I, R, B <: AMDGPU.ROCBackend}
    return KAPreconditioners.build_coarse_solver(
        matrix, KAPreconditioners.matrix_backend(matrix)
    )
end

end
