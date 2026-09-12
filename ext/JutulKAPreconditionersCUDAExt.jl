module JutulKAPreconditionersCUDAExt

using Jutul
using Jutul.KAPreconditioners
using Jutul: StaticSparsityMatrixCSR
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using KernelAbstractions
using LinearAlgebra

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

end
