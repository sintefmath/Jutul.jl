export StaticSparsityMatrixCSR, colvals, static_sparsity_sparse

struct StaticSparsityMatrixCSR{Tv,Ti<:Integer} <: SparseArrays.AbstractSparseMatrix{Tv,Ti}
    At::SparseMatrixCSC{Tv, Ti}
    nthreads::Int64
    minbatch::Int64
end

Base.size(S::StaticSparsityMatrixCSR) = reverse(size(S.At))
Base.getindex(S::StaticSparsityMatrixCSR, I::Integer, J::Integer) = S.At[J, I]
SparseArrays.nnz(S::StaticSparsityMatrixCSR) = nnz(S.At)
SparseArrays.nonzeros(S::StaticSparsityMatrixCSR) = nonzeros(S.At)
Base.isstored(S::StaticSparsityMatrixCSR, I::Integer, J::Integer) = Base.isstored(S.At, J, I)
SparseArrays.nzrange(S::StaticSparsityMatrixCSR, row::Integer) = SparseArrays.nzrange(S.At, row)

colvals(S::StaticSparsityMatrixCSR) = SparseArrays.rowvals(S.At)

function LinearAlgebra.mul!(y::AbstractVector, A::StaticSparsityMatrixCSR, x::AbstractVector, α::Number, β::Number)
    At = A.At
    n = size(y, 1)
    size(A, 1) == size(x, 1) || throw(DimensionMismatch())
    size(A, 2) == n || throw(DimensionMismatch())
    mb = max(n ÷ A.nthreads, 1)
    @batch minbatch = mb for row in 1:n
        v = zero(eltype(y))
        @inbounds for nz in nzrange(A, row)
            col = At.rowval[nz]
            v += At.nzval[nz]*x[col]
        end
        @inbounds y[row] = α*v + β*y[row]
    end
    return y
end

function static_sparsity_sparse(I, J, V, m = maximum(I), n = maximum(J); kwarg...)
    A = sparse(J, I, V, n, m)
    return StaticSparsityMatrixCSR(A; kwarg...)
end

function StaticSparsityMatrixCSR(A::SparseMatrixCSC; nthreads = Threads.nthreads(), minbatch = 1000)
    return StaticSparsityMatrixCSR(A, nthreads, minbatch)
end

function StaticSparsityMatrixCSR(m, n, rowptr, cols, nzval; kwarg...)
    # @info "Setting up" m n rowptr cols nzval
    At = SparseMatrixCSC(n, m, rowptr, cols, nzval)
    return StaticSparsityMatrixCSR(At; kwarg...)
end
