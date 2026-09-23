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

function KAPreconditioners.build_vendor_ilu(
        A::StaticSparsityMatrixCSR{Tv, Ti, V, I, R, B}
    ) where {Tv, Ti <: Integer, V, I, R, B <: AMDGPU.ROCBackend}
    scalar_type = KAPreconditioners.matrix_scalar_type(Tv)
    scalar_type <: ROCSPARSEValue || throw(
        ArgumentError(
            "AMDGPU VendorILU does not support matrix scalar type $scalar_type"
        )
    )
    factor_values = copy(A.nzval)
    rowptr = copy(A.rowptr)
    colval = copy(A.colval)
    factor = if Tv <: Number
        ROCSparseMatrixCSR{Tv, Ti}(
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
        ROCSparseMatrixBSR{scalar_type, Ti}(
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
    KAPreconditioners.refactor_vendor_ilu!(factor)
    return factor, factor_values
end

function KAPreconditioners.refactor_vendor_ilu!(
        factor::Union{ROCSparseMatrixCSR, ROCSparseMatrixBSR}
    )
    rocSPARSE.ilu0!(factor, 'O')
    return factor
end

function KAPreconditioners.vendor_ilu_factor_storage_bytes(
        factor::Union{ROCSparseMatrixCSR, ROCSparseMatrixBSR}
    )
    return sizeof(eltype(factor.rowPtr)) * length(factor.rowPtr) +
        sizeof(eltype(factor.colVal)) * length(factor.colVal)
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
