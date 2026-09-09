
struct ParallelILUFactorCSR{N, T, A} <: AbstractILUFactorization
    factors::NTuple{N, T}
    active::NTuple{N, A}
    threads::Symbol
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
    return ParallelILUFactorCSR{N, T, VT}(F, active, A.thread_type)
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
    function F(i)
        f = ilu0_csr(A, active = active[i])
        f::eltype(factors)
        factors[i] = f
    end
    threaded_loop(F, N, A.thread_type)
    return factors
end


function ilu0_csr(A::StaticSparsityMatrixCSR, active::NTuple)
    factor = ParallelILUFactorCSR(A, active)
    return factor
end

function update_factor!(LU::ParallelILUFactorCSR, A, i)
    return ilu0_csr!(LU.factors[i], A)
end

function apply_factor!(x, LU::ParallelILUFactorCSR, b, i)
    return ldiv!(x, LU.factors[i], b)
end

function ilu0_csr!(LU::ParallelILUFactorCSR{N, T, G}, A::StaticSparsityMatrixCSR) where {N, T, G}
    F(i) = update_factor!(LU, A, i)
    threaded_loop(F, N, LU.threads)
    return LU
end

function ldiv!(x::AbstractVector, LU::ParallelILUFactorCSR{N, T, A}, b::AbstractVector) where {N, T, A}
    F(i) = apply_factor!(x, LU, b, i)
    threaded_loop(F, N, LU.threads)
    return x
end

