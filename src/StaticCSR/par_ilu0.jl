
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

function ilu0_csr(A::StaticSparsityMatrixCSR, partition::V) where {V<:AbstractVector}
    N = maximum(partition)
    @assert minimum(partition) > 0
    @assert length(partition) == size(A, 1)
    active = tuple(map(i -> findall(isequal(i), partition), 1:N)...)
    active::NTuple
    factor = ParallelILUFactorCSR(A, active)
    return factor
end


function ilu_initial_setup_par!(factors, A, active, N)
    Threads.@threads :static for i in 1:N
        f = ilu0_csr(A, active = active[i])
        f::eltype(factors)
        factors[i] = f
    end
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
