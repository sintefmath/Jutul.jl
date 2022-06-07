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

