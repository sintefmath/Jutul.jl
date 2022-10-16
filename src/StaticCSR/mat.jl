struct StaticSparsityMatrixCSR{Tv,Ti<:Integer} <: SparseArrays.AbstractSparseMatrix{Tv,Ti}
    At::SparseMatrixCSC{Tv, Ti}
    nthreads::Int64
    minbatch::Int64
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
    for row in axes(M, 1)
        for pos in nzrange(M, row)
            col = columns[pos]
            nz[pos] = rowcol_prod(A, B, row, col)
        end
    end
end

function in_place_mat_mat_mul!(M::CSC, A::CSR, B::CSC) where {CSR<:StaticSparsityMatrixCSR, CSC<:SparseMatrixCSC}
    rows = rowvals(M)
    nz = nonzeros(M)
    for col in axes(M, 2)
        for pos in nzrange(M, col)
            row = rows[pos]
            nz[pos] = rowcol_prod(A, B, row, col)
        end
    end
end

function rowcol_prod(A::StaticSparsityMatrixCSR, B::SparseMatrixCSC, row, col)
    # We know both that this product is nonzero
    # First matrix, iterate over columns
    A_range = nzrange(A, row)
    nz_A = nonzeros(A)
    n_col = length(A_range)
    columns = colvals(A)
    new_column(pos) = columns[A_range[pos]]

    # Second matrix, iterate over row
    B_range = nzrange(B, col)
    nz_B = nonzeros(B)
    n_row = length(B_range)
    rows = rowvals(B)
    new_row(pos) = rows[B_range[pos]]

    # Initialize
    pos_A = pos_B = 1
    current_col = new_column(pos_A)
    current_row = new_row(pos_B)
    v = zero(eltype(A))
    while true# it < 100
        if current_row == current_col
            v += nz_A[pos_A]*nz_B[pos_B]
            increment_col = increment_row = true
        else
            increment_col = current_col < current_row
            increment_row = !increment_col
        end

        if increment_row
            pos_B += 1
            if pos_B > n_row
                break
            end
            current_row = new_row(pos_B)
        end
        if increment_col
            pos_A += 1
            if pos_A > n_col
                break
            end
            current_col = new_column(pos_A)
        end
    end
    return v
end

function rowcol_prod(A::SparseMatrixCSC, B::StaticSparsityMatrixCSR, row, col)
    v = zero(eltype(A))
    error()
end


function coarse_product!(C::T, A::T, R::T) where T<:StaticSparsityMatrixCSR
    n, m = size(C)
    # R = P'
    nz_c = nonzeros(C)
    cols_c = colvals(C)
    mb = minbatch(C, n)
    # @batch minbatch=mb for i in 1:n
    for i in 1:n
        @inbounds for j_p in nzrange(C, i)
            j = cols_c[j_p]
            nz_c[j_p] = compute_A_c_ij(A, R, i, j)
        end
    end
    return C
end

function compute_A_c_ij(A, R, i, j)
    # Compute A_c = R*A*P = R*A*R' in-place by
    # (R*A*P)_ij = sum_l R_il sum_k A_lk A_kj
    nz_a = nonzeros(A)
    nz_r = nz_p = nonzeros(R)
    cols_a = colvals(A)
    rows_p = cols_r = colvals(R)
    P_rng = nzrange(R, j)
    M = length(P_rng)

    v = 0.0
    @inbounds for l_p in nzrange(R, i)
        l = cols_r[l_p]
        # Now sum over the others two matrices
        A_rng = nzrange(A, l)
        # Loop over both P (=R') and A
        acc = 0.0
        A_pos = P_pos = 1
        N = length(A_rng)
        @inbounds while A_pos <= N && P_pos <= M
            # A
            p_A = A_rng[A_pos]
            kA = cols_a[p_A]
            # P
            p_P = P_rng[P_pos]
            kP = rows_p[p_P]
            if kA == kP
                @inbounds acc += nz_a[p_A]*nz_p[p_P]
                # Increment both counters
                A_pos += 1
                P_pos += 1
            elseif kA < kP
                A_pos += 1
            else
                P_pos += 1
            end
        end
        R_il = nz_r[l_p]
        v += R_il*acc
    end
    return v
end

