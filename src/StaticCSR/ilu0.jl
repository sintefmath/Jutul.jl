import LinearAlgebra.ldiv!, LinearAlgebra.\, SparseArrays.nnz

export ldiv!

function keep(col, row, lower)
    if lower
        return col < row
    else
        return col > row
    end
end

function fixed_block(A::StaticSparsityMatrixCSR{Tv, Ti}, active = axes(A, 1), order = axes(A, 1); lower::Bool = true) where {Tv, Ti}
    cols = colvals(A)
    vals = nonzeros(A)
    n, m = size(A)
    rowptr = zeros(Ti, n+1)
    offset = 1
    rowptr[1] = offset
    @inbounds for row in 1:n
        ctr = 0
        if insorted(row, active)
            for i in nzrange(A, order[row])
                @inbounds col = cols[i]
                if keep(order[col], row, lower) && insorted(col, active)
                    ctr += 1
                end
            end
        end
        offset = offset + ctr
        rowptr[row+1] = offset
    end
    # Next, we actually fill in now that we know the sizes
    sub_cols = zeros(Ti, offset-1)
    sub_vals = zeros(Tv, offset-1)

    map = zeros(Ti, offset-1)
    index = 1
    for row in 1:n
        if insorted(row, active)
            @inbounds for i in nzrange(A, row)
                col = order[cols[i]]
                @inbounds if keep(col, row, lower) && insorted(col, active)
                    map[index] = i
                    sub_cols[index] = col
                    index += 1
                end
            end
        end
    end
    B = StaticSparsityMatrixCSR(n, m, rowptr, sub_cols, sub_vals, nthreads = A.nthreads, minbatch = A.minbatch)
    update_values!(B, A, map)
    return (B, map)
end

function diagonal_block(A::StaticSparsityMatrixCSR{Tv, Ti}, active = axes(A, 1), order = axes(A, 1)) where {Tv, Ti}
    n = length(active)
    N = size(A, 1)
    out = zeros(Tv, n)
    pos = zeros(Ti, n)
    cols = colvals(A)
    vals = nonzeros(A)
    for (i, row) in enumerate(active)
        for k in nzrange(A, row)
            col = order[cols[k]]
            if col == row
                out[i] = vals[k]
                pos[i] = k
                break
            else
                @assert col < row "Diagonal must be present in sparsity pattern."
            end
        end
    end
    if n < N
        out = SparseVector(N, collect(active), out)
    else
        @assert n == N
    end
    return (out, pos)
end

function update_values!(B, A, mapping)
    vals = nonzeros(A)
    if issparse(B)
        bvals = nonzeros(B)
    else
        bvals = B
    end
    inner_update!(bvals, vals, mapping)
    return B
end

@inline function inner_update!(bvals, vals, mapping)
    @inbounds for (i, m) in enumerate(mapping)
        bvals[i] = vals[m]
    end
end

@inline function process_partial_row!(nz, pos, cols, A, k, A_ik)
    @inbounds for l_j in pos
        j = cols[l_j]
        A_kj = A[k, j]
        nz[l_j] -= A_ik*A_kj
    end
end

function ilu0_factor!(L, U, D, A, active = axes(A, 1))
    A = missing
    cols_l = colvals(L)
    nz_l = nonzeros(L)

    cols_u = colvals(U)
    nz_u = nonzeros(U)
    for i in active
        l_pos = nzrange(L, i)
        u_pos = nzrange(U, i)
        l_start = 1
        @inbounds for l_i in l_pos
            k = cols_l[l_i]
            A_kk = D[k]
            A_ik = nz_l[l_i]*inv(A_kk)
            # Put it back as inverted
            nz_l[l_i] = A_ik
            if A_ik != zero(eltype(D))
                # Do the remainder of the row: First the part inside L, diagonal, and then the part in U
                rem_l_pos = @view l_pos[l_start+1:end]
                # In the following:
                # k = 1:(i-1)
                # and we loop over
                # j = (k+1):n
                # This implies that j > k. Accesses to A[k, j] are
                # then only applied to the upper part of the original
                # matrix.
                process_partial_row!(nz_l, rem_l_pos, cols_l, U, k, A_ik)
                D[i] -= A_ik*U[k, i]
                process_partial_row!(nz_u, u_pos, cols_u, U, k, A_ik)
            end
            l_start += 1
        end
    end
    for i in active
        @inbounds D[i] = inv(D[i])
    end
end

@inline Base.@propagate_inbounds function apply_diagonal_inverse(D::SparseVector, global_index, local_index, v)
    return nonzeros(D)[local_index]*v
end

@inline Base.@propagate_inbounds function apply_diagonal_inverse(D, global_index, local_index, v)
    return D[global_index]*v
end

@inline apply_diagonal_inverse(::Nothing, global_index, local_index, v) = v

@inline function invert_row!(x, M, D, b, row, local_index)
    col = colvals(M)
    nz = nonzeros(M)

    @inbounds v = b[row]
    @inbounds for j in nzrange(M, row)
        k = col[j]
        v -= nz[j]*x[k]
    end
    @inbounds x[row] = apply_diagonal_inverse(D, row, local_index, v)
end

function forward_substitute!(x, M, b, order = 1:length(b), D = nothing)
    @inbounds for (local_index, i) in enumerate(order)
        invert_row!(x, M, D, b, i, local_index)
    end
    return x
end

function backward_substitute!(x, M, b, order = 1:length(b), D = nothing)
    n = length(order)
    @inbounds for (i, local_index) in zip(Iterators.reverse(order), n:-1:1)
        invert_row!(x, M, D, b, i, local_index)
    end
    return x
end

export ilu_solve!
function ilu_solve!(x, L, U, D, b, active = eachindex(b), order = eachindex(b))
    forward_substitute!(x, L, b, active)
    out = backward_substitute!(x, U, x, active, D)
    @. out[order] = out
    return out
end

abstract type AbstractILUFactorization end

struct ILUFactorCSR{Mat_t, Diag_t, Map_t, Act_t, Order_t} <: AbstractILUFactorization
    L::Mat_t
    U::Mat_t
    D::Diag_t
    active::Act_t
    L_map::Map_t
    U_map::Map_t
    D_map::Map_t
    order::Order_t
    function ILUFactorCSR(L::Mt, U::Mt, D::Dt, L_map::Lt, U_map::Lt, D_map::Lt; active = axes(L, 1), order = axes(L, 1)) where {Mt, Dt, Lt}
        return new{Mt, Dt, Lt, typeof(active), typeof(order)}(L, U, D, active, L_map, U_map, D_map, order)
    end
end

Base.eltype(ilu::ILUFactorCSR{M, D, U, A}) where {M, D, U, A} = eltype(M)
function Base.show(io::IO, t::MIME"text/plain", ilu::ILUFactorCSR)
    n, m = size(ilu.L)
    println(io, "$ILUFactorCSR of size ($n, $m) with eltype $(eltype(ilu))")
    println(io, "L: $(nnz(ilu.L)) nonzeros")
    println(io, "U: $(nnz(ilu.U)) nonzeros")
end

export ilu0_csr, ilu0_csr!
function ilu0_csr(A::StaticSparsityMatrixCSR; active = axes(A, 1), order = axes(A, 1))
    n, m = size(A)
    @assert n == m
    L, ml = fixed_block(A, active, order, lower = true)
    U, mu = fixed_block(A, active, order, lower = false)
    D, md = diagonal_block(A, active, order)
    ilu0_factor!(L, U, D, A, active)
    return ILUFactorCSR(L, U, D, ml, mu, md, active = active, order = order)
end

function ilu0_csr!(LU::ILUFactorCSR, A::StaticSparsityMatrixCSR)
    L, U, D = LU.L, LU.U, LU.D
    update_values!(L, A, LU.L_map)
    update_values!(U, A, LU.U_map)
    update_values!(D, A, LU.D_map)
    ilu0_factor!(L, U, D, A, LU.active)
    return LU
end


function ldiv!(x::AbstractVector, LU::ILUFactorCSR, b::AbstractVector)
    x = ilu_solve!(x, LU.L, LU.U, LU.D, b, LU.active, LU.order)
    return x
end

function ldiv!(LU::AbstractILUFactorization, b)
    return ldiv!(b, LU, b)
end

function \(LU::AbstractILUFactorization, b)
    x = similar(b)
    return ldiv!(x, LU, b)
end
