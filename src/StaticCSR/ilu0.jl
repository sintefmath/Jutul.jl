import LinearAlgebra.ldiv!, LinearAlgebra.\, SparseArrays.nnz

function keep(col, row, lower)
    if lower
        return col < row
    else
        return col > row
    end
end

function fixed_block(A::StaticSparsityMatrixCSR{Tv, Ti}, active = 1:size(A, 1); lower::Bool = true) where {Tv, Ti}
    cols = colvals(A)
    vals = nonzeros(A)
    n, m = size(A)
    rowptr = zeros(Ti, n+1)
    offset = 1
    rowptr[1] = offset
    @inbounds for row in 1:n
        ctr = 0
        if insorted(row, active)
            for i in nzrange(A, row)
                @inbounds col = cols[i]
                if keep(col, row, lower) && insorted(col, active)
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
        ctr = 0
        if insorted(row, active)
            @inbounds for i in nzrange(A, row)
                col = cols[i]
                @inbounds if keep(col, row, lower) && insorted(col, active)
                    map[index] = i
                    sub_vals[index] = vals[i]
                    sub_cols[index] = col
                    index += 1
                end
            end
        end
    end
    B = StaticSparsityMatrixCSR(n, m, rowptr, sub_cols, sub_vals)
    return (B, map)
end

function diagonal_block(A::StaticSparsityMatrixCSR{Tv, Ti}, active = 1:size(A, 1)) where {Tv, Ti}
    n = length(active)
    N = size(A, 1)
    out = zeros(Tv, n)
    pos = zeros(Ti, n)
    cols = colvals(A)
    vals = nonzeros(A)
    for (i, row) in enumerate(active)
         for k in nzrange(A, row)
            col = cols[k]
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
        out = SparseVector(N, active, out)
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
    @inbounds for (i, m) in enumerate(mapping)
         bvals[i] = vals[m]
    end
    return B
end

function process_partial_row!(nz, pos, cols, K, k, A_ik)
    @inbounds for l_j in pos
        j = cols[l_j]
        A_kj = K[k, j]
        nz[l_j] -= A_ik*A_kj
    end
end

function ilu0_factor!(L, U, D, A; active = 1:size(A, 1))
    n, m = size(L)
    cols_l = colvals(L)
    nz_l = nonzeros(L)

    cols_u = colvals(U)
    nz_u = nonzeros(U)
    for i in active
        l_pos = nzrange(L, i)
        u_pos = nzrange(U, i)
        @inbounds for (l_start, l_i) in enumerate(l_pos)
            k = cols_l[l_i]
            A_kk = D[k]
            A_ik = nz_l[l_i]*inv(A_kk)
            # Put it back as inverted
            nz_l[l_i] = A_ik
            # Do the remainder of the row: First the part inside L, diagonal, and then the part in U
            rem_l_pos = @view l_pos[l_start+1:end]
            K = U
            process_partial_row!(nz_l, rem_l_pos, cols_l, K, k, A_ik)
            D[i] -= A_ik*K[k, i]
            process_partial_row!(nz_u, u_pos, cols_u, K, k, A_ik)
        end
    end
    @inbounds for i in active
        D[i] = inv(D[i])
    end
end
Base.@propagate_inbounds diagonal_inverse(D, i) = D[i]
diagonal_inverse(::Nothing, i) = 1.0

function trisolve!(x, M, b; order = 1:length(b), D = nothing)
    # Note: Already inverted if provided
    col = colvals(M)
    nz = nonzeros(M)
    @inbounds for i in order
        v = b[i]
        @inbounds for j in nzrange(M, i)
            k = col[j]
            v -= nz[j]*x[k]
        end
        x[i] = diagonal_inverse(D, i)*v
    end
    return x
end

export ilu_solve!
function ilu_solve!(x, L, U, D, b; active = 1:length(b))
    trisolve!(x, L, b, order = active)
    return trisolve!(x, U, x, order = Iterators.reverse(active), D = D)
end

abstract type AbstractILUFactorization end

struct ILUFactorCSR{Mat_t, Diag_t, Map_t, Act_t} <: AbstractILUFactorization
    L::Mat_t
    U::Mat_t
    D::Diag_t
    active::Act_t
    L_map::Map_t
    U_map::Map_t
    D_map::Map_t
    function ILUFactorCSR(L::Mt, U::Mt, D::Dt, L_map::Lt, U_map::Lt, D_map::Lt; active = 1:size(L, 1)) where {Mt, Dt, Lt}
        return new{Mt, Dt, Lt, typeof(active)}(L, U, D, active, L_map, U_map, D_map)
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
function ilu0_csr(A::StaticSparsityMatrixCSR; active = 1:size(A, 1))
    n, m = size(A)
    @assert n == m
    L, ml = fixed_block(A, active, lower = true)
    U, mu = fixed_block(A, active, lower = false)
    D, md = diagonal_block(A, active)
    ilu0_factor!(L, U, D, A, active = active)
    return ILUFactorCSR(L, U, D, ml, mu, md, active = active)
end

function ilu0_csr!(LU::ILUFactorCSR, A::StaticSparsityMatrixCSR)
    L, U, D = LU.L, LU.U, LU.D
    update_values!(L, A, LU.L_map)
    update_values!(U, A, LU.U_map)
    update_values!(D, A, LU.D_map)
    ilu0_factor!(L, U, D, A, active = LU.active)
    return LU
end


function ldiv!(x::AbstractVector, LU::ILUFactorCSR, b::AbstractVector)
    x = ilu_solve!(x, LU.L, LU.U, LU.D, b, active = LU.active)
    return x
end

function ldiv!(LU::AbstractILUFactorization, b)
    return ldiv!(b, LU, b)
end

function \(LU::AbstractILUFactorization, b)
    x = similar(b)
    return ldiv!(x, LU, b)
end

struct ParallelILUFactorCSR{N, T, A} <: AbstractILUFactorization
    factors::NTuple{N, T}
    active::NTuple{N, A}
end

Base.eltype(ilu::ParallelILUFactorCSR) = Base.eltype(first(ilu.factors))

function Base.show(io::IO, t::MIME"text/plain", ilu::ParallelILUFactorCSR)
    n = length(ilu.factors)
    println(io, "ParallelILUFactorCSR with $n threads:")
    for (i, f) in enumerate(ilu.factors)
        act = ilu.active[i]
        na = length(act)
        print(io, "Subdomain $i: $na elements: [")
        lim = 25
        for i = 1:(lim-1)
            print(io, "$(act[i]), ")
        end
        print(io, act[lim])
        if na > lim
            println(io, ", ... ]")
        else
            println(io, "]")
        end
    end
end

function ParallelILUFactorCSR(A::StaticSparsityMatrixCSR{Tv, Ti}, active::Tuple) where {Tv, Ti}
    M = StaticSparsityMatrixCSR{Tv, Ti}
    N = length(active)
    VT = Vector{Ti}
    AT = eltype(active)
    if N == 1
        Mt = Vector{Tv}
    else
        Mt = SparseVector{Tv, Ti}
    end
    T = ILUFactorCSR{M, Mt, VT, AT}
    factors = Vector{T}(undef, N)
    ilu_initial_setup_par!(factors, A, active, N)
    F = tuple(factors...)
    F::NTuple{N, T}
    return ParallelILUFactorCSR{N, T, VT}(F, active)
end

function ilu_initial_setup_par!(factors, A, active, N)
    Threads.@threads for i in 1:N
        f = ilu0_csr(A, active = active[i])
        f::eltype(factors)
        factors[i] = f
    end
end

function ilu0_csr(A::StaticSparsityMatrixCSR, partition::V) where {V<:AbstractVector}
    N = maximum(partition)
    @assert minimum(partition) > 0
    @assert length(partition) == size(A, 1)
    active = tuple(map(i -> findall(isequal(i), partition), 1:N)...)
    active::NTuple
    factor = ParallelILUFactorCSR(A, active)
    return factor
end

function ilu0_csr(A::StaticSparsityMatrixCSR, active::NTuple)
    factor = ParallelILUFactorCSR(A, active)
    return factor
end

update_factor!(LU::ParallelILUFactorCSR, A, i) = ilu0_csr!(LU.factors[i], A)
apply_factor!(x, LU::ParallelILUFactorCSR, b, i) = ldiv!(x, LU.factors[i], b)

function ilu0_csr!(LU::ParallelILUFactorCSR{N, T, G}, A::StaticSparsityMatrixCSR) where {N, T, G}
    Threads.@threads for i in 1:N
        update_factor!(LU, A, i)
    end
    return LU
end

function ldiv!(x::AbstractVector, LU::ParallelILUFactorCSR{N, T, A}, b::AbstractVector) where {N, T, A}
    Threads.@threads for i in 1:N
        apply_factor!(x, LU, b, i)
    end
    return x
end
