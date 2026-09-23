module JutulCUDSSExt

using Jutul
using Jutul.KAPreconditioners
using Jutul: StaticSparsityMatrixCSR
using CUDA
using CUDSS
using LinearAlgebra

mutable struct CUDSSSparseLUFactor{F, M, V, R, C}
    solver::F
    matrix::M
    rhs::V
    solution::V
    n::Int
    pattern_rowptr::Vector{Int}
    pattern_colval::Vector{Int}
    source_rowptr::R
    source_colval::C
end

function KAPreconditioners.setup_preferred_sparse_lu(
        A::StaticSparsityMatrixCSR{
            Tv, Ti, V, I, R, B,
        }
    ) where {
        Tv <: Union{Float32, Float64, ComplexF32, ComplexF64},
        Ti <: Union{Int32, Int64}, V, I, R, B <: CUDA.CUDABackend,
    }
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("sparse LU requires a square matrix"))
    matrix = CUDA.CUSPARSE.CuSparseMatrixCSR{Tv, Ti}(
        A.rowptr, A.colval, A.nzval, size(A)
    )
    solver = lu(matrix)
    factor = CUDSSSparseLUFactor(
        solver, matrix, CUDA.zeros(Tv, n), CUDA.zeros(Tv, n), n,
        Int.(Array(A.rowptr)), Int.(Array(A.colval)), A.rowptr, A.colval
    )
    return KAPreconditioners.SparseLU(factor)
end

function KAPreconditioners.resetup_sparse_lu!(
        S::KAPreconditioners.SparseLU{<:CUDSSSparseLUFactor},
        A::StaticSparsityMatrixCSR
    )
    KAPreconditioners.sparse_lu_same_pattern(S, A) ||
        throw(ArgumentError("sparse LU resetup requires the same CSR pattern"))
    F = S.factorization
    copyto!(F.matrix.nzVal, A.nzval)
    lu!(F.solver, F.matrix)
    return S
end

function LinearAlgebra.ldiv!(
        x, S::KAPreconditioners.SparseLU{<:CUDSSSparseLUFactor}, b
    )
    F = S.factorization
    length(x) == F.n && length(b) == F.n || throw(DimensionMismatch())
    copyto!(F.rhs, b)
    ldiv!(F.solution, F.solver, F.rhs)
    copyto!(x, F.solution)
    return x
end

end
