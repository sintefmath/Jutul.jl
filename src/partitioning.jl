export LinearPartitioner, MetisPartitioner
struct LinearPartitioner <: JutulPartitioner

end

function partition(::LinearPartitioner, A, m)
    n, l = size(A)
    @assert n == l
    return partition_linear(m, n)
end

function partition_linear(m, n)
    partition = zeros(Int64, n);
    for i in eachindex(partition)
        partition[i] = ceil(i / (n/m))
    end
    return partition
end

function partition_to_lookup(partition::AbstractVector, N = maximum(partition))
    return tuple(map(i -> findall(isequal(i), partition), 1:N)...)
end

function generate_lookup(partitioner, A, n)
    p = partition(partitioner, A, n)
    return partition_to_lookup(p, n)
end

struct MetisPartitioner <: JutulPartitioner
    algorithm::Symbol
    MetisPartitioner(m = :KWAY) = new(m)
end

function partition(mp::MetisPartitioner, A, m)
    return Metis.partition(generate_graph(A), m, alg = mp.algorithm)
end

metis_strength(F) = F
function metis_strength(F::AbstractMatrix)
    s = zero(eltype(F))
    n, m = size(F)
    for i = 1:min(n, m)
        s += F[i, i]
    end
    return s
end

function generate_graph(A::SparseMatrixCSC)
    n, m = size(A)
    @assert n == m
    i, j, v = findnz(A)
    V = map(metis_strength, v)
    I = vcat(i, j)
    J = vcat(j, i)
    V = vcat(V, V)
    M = sparse(I, J, V, n, n)
    return Metis.graph(M, check_hermitian = false)
end

generate_graph(A::StaticSparsityMatrixCSR) = generate_graph(A.At)
