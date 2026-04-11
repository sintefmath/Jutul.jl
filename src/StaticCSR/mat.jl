struct StaticSparsityMatrixCSR{Tv,Ti<:Integer,V<:AbstractVector{Tv},I<:AbstractVector{Ti}} <: SparseArrays.AbstractSparseMatrix{Tv,Ti}
    At::SparseMatrixCSC{Tv, Ti}
    nzval::V
    colval::I
    rowptr::Vector{Ti}
    nthreads::Int
    minbatch::Int
    m::Int
    n::Int
    function StaticSparsityMatrixCSR(A_t::SparseMatrixCSC{Tv, Ti}; nthreads = Threads.nthreads(), minbatch = 1000) where {Tv, Ti}
        nz = nonzeros(A_t)
        cv = SparseArrays.rowvals(A_t)
        rp = A_t.colptr
        nrows, ncols = reverse(size(A_t))
        return new{Tv, Ti, typeof(nz), typeof(cv)}(A_t, nz, cv, rp, nthreads, minbatch, nrows, ncols)
    end
    function StaticSparsityMatrixCSR(nzval::V, colval::I, rowptr::Vector{Ti}, m::Int, n::Int;
            nthreads = Threads.nthreads(), minbatch = 1000) where {Tv, Ti, V<:AbstractVector{Tv}, I<:AbstractVector{Ti}}
        # Constructor without At - for device arrays where SparseMatrixCSC is not available
        At_dummy = spzeros(Tv, Ti, 0, 0)
        return new{Tv, Ti, V, I}(At_dummy, nzval, colval, rowptr, nthreads, minbatch, m, n)
    end
end

Base.size(S::StaticSparsityMatrixCSR) = (S.m, S.n)
Base.getindex(S::StaticSparsityMatrixCSR, I::Integer, J::Integer) = S.At[J, I]
SparseArrays.nnz(S::StaticSparsityMatrixCSR) = length(S.nzval)
SparseArrays.nonzeros(S::StaticSparsityMatrixCSR) = S.nzval
Base.isstored(S::StaticSparsityMatrixCSR, I::Integer, J::Integer) = Base.isstored(S.At, J, I)
function SparseArrays.nzrange(S::StaticSparsityMatrixCSR, row::Integer)
    return S.rowptr[row]:(S.rowptr[row+1]-1)
end

function SparseArrays.findnz(S::StaticSparsityMatrixCSR)
    J, I, V = findnz(S.At)
    return (I, J, V)
end

colvals(S::StaticSparsityMatrixCSR) = S.colval

function LinearAlgebra.mul!(y::AbstractVector, A::StaticSparsityMatrixCSR, x::AbstractVector, α::Number, β::Number)
    n = size(y, 1)
    size(A, 2) == size(x, 1) || throw(DimensionMismatch())
    size(A, 1) == n || throw(DimensionMismatch())
    mb = max(n ÷ nthreads(A), minbatch(A))
    nzval = nonzeros(A)
    cv = colvals(A)
    rp = A.rowptr
    if β == 0
        csr_mul_add!(y, nzval, cv, rp, x, n, mb, α, Val(false))
    else
        if β != 1
            rmul!(y, β)
        end
        csr_mul_add!(y, nzval, cv, rp, x, n, mb, α, Val(true))
    end
    return y
end

function csr_mul_add!(y::AbstractVector{Ty}, nzval, rowval, colptr, x, n, mb, α, ::Val{do_increment}) where {do_increment, Ty}
    @batch minbatch = mb for row in 1:n
        v = zero(Ty)
        @inbounds start = colptr[row]
        @inbounds stop = colptr[row+1]-1
        @inbounds for nz in start:stop
            col = rowval[nz]
            A_ij = nzval[nz]
            x_j = x[col]
            v = internal_muladd(A_ij, x_j, v)
        end
        if do_increment
            @inbounds y[row] += α*v
        else
            @inbounds y[row] = α*v
        end
    end
end

@inline function internal_muladd(A_ij, x_j, v)
    return v + A_ij*x_j
end
@inline function internal_muladd(A_ij, x_j::SVector, v::SVector)
    return muladd(A_ij, x_j, v)
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
