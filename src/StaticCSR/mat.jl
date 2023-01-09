struct StaticSparsityMatrixCSR{Tv,Ti<:Integer} <: SparseArrays.AbstractSparseMatrix{Tv,Ti}
    At::SparseMatrixCSC{Tv, Ti}
    nthreads::Int
    minbatch::Int
    function StaticSparsityMatrixCSR(A_t::SparseMatrixCSC{Tv, Ti}; nthreads = Threads.nthreads(), minbatch = 1000) where {Tv, Ti}
        return new{Tv, Ti}(A_t, nthreads, minbatch)
    end
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
    size(A, 2) == size(x, 1) || throw(DimensionMismatch())
    size(A, 1) == n || throw(DimensionMismatch())
    mb = max(n ÷ nthreads(A), minbatch(A))
    if β != 1
        β != 0 ? rmul!(y, β) : fill!(y, zero(eltype(y)))
    end
    @batch minbatch = mb for row in 1:n
        v = zero(eltype(y))
        @inbounds for nz in nzrange(A, row)
            col = At.rowval[nz]
            v += At.nzval[nz]*x[col]
        end
        @inbounds y[row] += α*v
    end
    return y
end

nthreads(A::StaticSparsityMatrixCSR) = A.nthreads
minbatch(A::StaticSparsityMatrixCSR) = A.minbatch

function static_sparsity_sparse(I, J, V, m = maximum(I), n = maximum(J); kwarg...)
    A = sparse(J, I, V, n, m)
    return StaticSparsityMatrixCSR(A; kwarg...)
end


function StaticSparsityMatrixCSR(m, n, rowptr, cols, nzval; kwarg...)
    # @info "Setting up" m n rowptr cols nzval
    At = SparseMatrixCSC(n, m, rowptr, cols, nzval)
    return StaticSparsityMatrixCSR(At; kwarg...)
end


function in_place_mat_mat_mul!(M::CSR, A::CSR, B::CSC) where {CSR<:StaticSparsityMatrixCSR, CSC<:SparseMatrixCSC}
    columns = colvals(M)
    nz = nonzeros(M)
    n = size(M, 1)
    mb = max(n ÷ nthreads(A), minbatch(A))
    @batch minbatch = mb for row in 1:n
        for pos in nzrange(M, row)
            @inbounds col = columns[pos]
            @inbounds nz[pos] = rowcol_prod(A, B, row, col)
        end
    end
end

function in_place_mat_mat_mul!(M::CSC, A::CSR, B::CSC) where {CSR<:StaticSparsityMatrixCSR, CSC<:SparseMatrixCSC}
    rows = rowvals(M)
    nz = nonzeros(M)
    n = size(M, 2)
    mb = max(n ÷ nthreads(A), minbatch(A))
    @batch minbatch = mb for col in 1:n
        for pos in nzrange(M, col)
            @inbounds row = rows[pos]
            @inbounds nz[pos] = rowcol_prod(A, B, row, col)
        end
    end
end

@inline function rowcol_prod(A::StaticSparsityMatrixCSR, B::SparseMatrixCSC, row, col)
    # We know both that this product is nonzero
    # First matrix, iterate over columns
    A_range = nzrange(A, row)
    nz_A = nonzeros(A)
    n_col = length(A_range)
    columns = colvals(A)
    @inline new_column(pos) = sparse_indirection(columns, A_range, pos)

    # Second matrix, iterate over row
    B_range = nzrange(B, col)
    nz_B = nonzeros(B)
    n_row = length(B_range)
    rows = rowvals(B)
    new_row(pos) = sparse_indirection(rows, B_range, pos)
    # Initialize
    pos_A = pos_B = 1
    current_col, A_idx = new_column(pos_A)
    current_row, B_idx = new_row(pos_B)
    v = zero(eltype(A))
    entries_remain = true
    while entries_remain
        delta = current_row - current_col
        if delta == 0
            @inbounds rv = nz_A[A_idx]
            @inbounds cv = nz_B[B_idx]
            v += rv*cv
            entries_remain = pos_A < n_col && pos_B < n_row
            if entries_remain
                pos_A += 1
                current_col, A_idx = new_column(pos_A)
                pos_B += 1
                current_row, B_idx = new_row(pos_B)
            end
        elseif delta > 0
            entries_remain = pos_A < n_col
            if entries_remain
                pos_A += 1
                current_col, A_idx = new_column(pos_A)
            end
        else
            entries_remain = pos_B < n_row
            if entries_remain
                pos_B += 1
                current_row, B_idx = new_row(pos_B)
            end
        end
    end
    return v
end

@inline function sparse_indirection(val, rng, pos)
    @inbounds ix = rng[pos]
    @inbounds v = val[ix]
    return (v, ix)
end

function rowcol_prod(A::SparseMatrixCSC, B::StaticSparsityMatrixCSR, row, col)
    v = zero(eltype(A))
    error()
end
