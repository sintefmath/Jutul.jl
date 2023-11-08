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

function partition(mp::MetisPartitioner, A::AbstractSparseMatrix, m; kwarg...)
    return partition(mp, generate_metis_graph(A), m; kwarg...)
end

function partition(mp::MetisPartitioner, g, m; alg = mp.algorithm, kwarg...)
    n = g.nvtxs
    if m == 1
        p = ones(Int, n)
    elseif n == m
        p = collect(1:m)
    else
        p = Metis.partition(g, m; alg = alg, kwarg...)
    end
    for x in 1:m
        @assert count(isequal(x), p) > 0 "Partitioning of array with $n entries into $m blocks failed: Block $x is empty."
    end
    return p
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

function generate_metis_graph(A::SparseMatrixCSC)
    n, m = size(A)
    @assert n == m
    i, j, v = findnz(A)
    V = map(metis_strength, v)
    V = metis_integer_weights(V)
    for i in eachindex(V)
        V[i] = clamp(V[i], 1, typemax(Int32))
    end
    I = vcat(i, j)
    J = vcat(j, i)
    V = vcat(V, V)
    M = sparse(I, J, V, n, n)
    return Metis.graph(M, check_hermitian = false, weights = true)
end

function metis_integer_weights(x::AbstractVector{<:Integer})
    return x
end

function metis_integer_weights(x::AbstractVector{<:AbstractFloat})
    mv = mean(x)*0.1;
    @. x = Int64(ceil(x / mv))
    return x
end

generate_metis_graph(A::StaticSparsityMatrixCSR) = generate_metis_graph(A.At)

function compress_partition(p::AbstractVector)
    up = sort!(unique(p))
    p_renum = copy(p)
    for i in eachindex(p)
        p_renum[i] = searchsortedfirst(up, p[i])
    end
    return p_renum
end

"""
    partition(N::AbstractMatrix, num_coarse, weights = ones(size(N, 2)); partitioner = MetisPartitioner(), groups = nothing, n = maximum(N), group_by_weights = false, buffer_group = true)

Partition based on neighborship (with optional groups kept contigious after
partitioning)
"""
function partition(N::AbstractMatrix, num_coarse, weights = ones(size(N, 2));
        partitioner = MetisPartitioner(),
        groups = nothing, n = maximum(N),
        group_by_weights = false,
        buffer_group = false
    )
    @assert size(N, 1) == 2
    @assert size(N, 2) == length(weights)
    weights::AbstractVector
    weights = metis_integer_weights(weights)
    num_coarse::Integer

    has_groups = !isnothing(groups)
    if has_groups && !group_by_weights
        # Some entries are clustered together and should not be divided. We
        # create a subgraph using a partition, and extend back afterwards.
        part = collect(1:n)
        for (i, group) in enumerate(groups)
            for g in group
                part[g] = n + i
            end
        end
        part = compress_partition(part)
        N = part[N]
        n_inner = maximum(part)
    else
        n_inner = n
    end
    if has_groups && group_by_weights
        maxv = 100*maximum(weights)
        for grp in groups
            for i in axes(N, 2)
                has_left = N[1, i] in grp
                has_right = N[2, i] in grp
                if buffer_group && (has_left || has_right)
                    interior = true
                elseif has_left && has_right
                    interior = true
                else
                    interior = false
                end
                if interior
                    weights[i] = maxv
                end
            end
        end
    end

    @assert num_coarse <= n_inner
    i = vec(N[1, :])
    j = vec(N[2, :])
    I = vcat(i, j)
    J = vcat(j, i)
    V = vcat(weights, weights)
    M = sparse(I, J, V, n_inner, n_inner)

    g = generate_metis_graph(M)
    p = partition(partitioner, g, num_coarse)
    if has_groups && !group_by_weights
        p = p[part]
    end
    return Int64.(p)
end


"""
    load_balanced_endpoint(block_index, nvals, nblocks)

Endpoint for interval `block_index` that subdivides `nvals` into `nblocks` in a
load balanced manner. This is done by adding one element to the first set of
blocks whenever possible.
"""
function load_balanced_endpoint(block_index, nvals, nblocks)
    # @assert nblocks <= nvals
    width = div(nvals, nblocks)
    # Gets added to the nblocks first elements
    remainder = mod(nvals, nblocks)
    # Count number of passed blocks that have an extra element
    passed_wide_blocks = min(block_index, remainder)
    return min(passed_wide_blocks + width*block_index, nvals)
end

"""
    load_balanced_interval(b, n, m)

Create UnitRange for block b ∈ [1, m] for interval of total length n
"""
function load_balanced_interval(b, n, m)
    start = load_balanced_endpoint(b-1, n, m) + 1
    stop = load_balanced_endpoint(b, n, m)
    return start:stop
end

"""
    setup_partitioner_hypergraph(N::Matrix{Int};
        num_nodes::Int = maximum(N),
        num_edges::Int = size(N, 2),
        node_weights::Vector{Int} = ones(Int, num_nodes),
        edge_weights::Vector{Int} = ones(Int, num_edges),
        groups = [Int[]]
    )

Set up a hypergraph structure for a given neighborship matrix. `N` should be a
matrix with two rows, with one pair of cells in each column. Optionally node and
edge weights can be provided. If a list of groups are provided, these nodes will
be accumulated together in the hypergraph.
"""
function setup_partitioner_hypergraph(N::Matrix{Int};
        num_nodes::Int = maximum(N),
        num_edges::Int = size(N, 2),
        node_weights::Vector{Int} = ones(Int, num_nodes),
        edge_weights::Vector{Int} = ones(Int, num_edges),
        groups = Vector{Vector{Int}}()
    )
    @assert size(N, 1) == 2
    @assert size(N, 2) == num_edges
    @assert length(edge_weights) == num_edges

    compressed_partition = collect(1:num_nodes)
    groups = merge_overlapping_graph_groups(groups)
    for (i, group) in enumerate(groups)
        for g in group
            compressed_partition[g] = num_nodes + i
        end
    end
    compressed_partition = Jutul.compress_partition(compressed_partition)
    N = compressed_partition[N]
    # Filter connections that are interior in a group
    keep = map(i -> N[1, i] != N[2, i], axes(N, 2))
    edge_indices = (1:num_edges)[keep]
    N = N[:, keep]
    num_nodes_compressed = maximum(compressed_partition)
    node_weights_compressed = zeros(Int, num_nodes_compressed)
    for (i, v) in enumerate(compressed_partition)
        node_weights_compressed[v] += node_weights[i]
    end
    @assert minimum(node_weights_compressed) > 0
    # Create map of all connections
    conn = Dict{Tuple{Int, Int}, Int}()
    for face in axes(N, 2)
        e = edge_indices[face]
        l, r = sort(N[:, face])
        c = (l, r)
        w_f = edge_weights[e]
        if haskey(conn, c)
            conn[c] += w_f
        else
            conn[c] = w_f
        end
    end
    I = Int[]
    J = Int[]
    pos = 1
    conn_pairs = keys(conn)
    num_edges_compressed = length(conn_pairs)
    edge_weights_compressed = zeros(Int, num_edges_compressed)

    N_new = zeros(Int, 2, num_edges_compressed)
    for (i, p) in enumerate(conn_pairs)
        l, r = p
        N_new[1, i] = l
        N_new[2, i] = r
        # I is cell
        push!(I, l)
        push!(I, r)

        # J is connection
        push!(J, i)
        push!(J, i)

        edge_weights_compressed[i] = conn[p]
    end
    V = ones(Int, length(I))
    A = sparse(I, J, V, num_nodes_compressed, num_edges_compressed)
    return (
        graph = A,
        node_weights = node_weights_compressed,
        edge_weights = edge_weights_compressed,
        neighbors = N_new,
        partition = compressed_partition,
        groups = groups
    )
end

"""
    partition_hypergraph(g, n::Int, partitioner = MetisPartitioner(); expand = true)

Partition a hypergraph from [setup_partitioner_hypergraph](@ref) using a given
partitioner. If the optional `expand` parameter is set to true the result will
be expanded to the full graph (i.e. where groups are not condensed).
"""
function partition_hypergraph(g::NamedTuple, n::Int, partitioner = MetisPartitioner(); expand = true)
    p = partition_hypergraph_implementation(g, n, partitioner)
    if expand
        p = p[g.partition]
    end
    return p
end

function partition_hypergraph(N::Matrix{Int}, n::Int, partitioner = MetisPartitioner(); expand = true, output_graph = false, kwarg...)
    g = setup_partitioner_hypergraph(N; kwarg...)
    p = partition_hypergraph(g, n, partitioner, expand = expand)
    if output_graph
        out = (p, g)
    else
        out = p
    end
end

function partition_hypergraph_implementation(hg, n, partitioner::JutulPartitioner)
    N = hg.neighbors
    n = size(hg.graph, 1)
    C = sparse(N[1, :], N[2, :], hg.edge_weights, n, n)
    C = C + C' + diagm(hg.node_weights)
    return partition(partitioner, C, n)
end

function partition_hypergraph_implementation(hg, n, mp::MetisPartitioner)
    g_metis = hypergraph_to_metis_format(hg)
    return Metis.partition(g_metis, n; alg = mp.algorithm)
end

function hypergraph_to_metis_format(hg)
    N = hg.neighbors
    n = size(hg.graph, 1)
    C = sparse(N[1, :], N[2, :], hg.edge_weights, n, n)
    C = C + C'
    g0 = Metis.graph(C, weights = true)
    w = similar(g0.adjwgt, n)
    @. w = hg.node_weights
    return Metis.Graph(g0.nvtxs, g0.xadj, g0.adjncy, w, g0.adjwgt)
end

function merge_overlapping_graph_groups(groups::Vector{Vector{Int}})
    # Merge groups that contain the same node
    if length(groups) == 0
        return groups
    else
        nc = maximum(maximum, groups)
        owned = zeros(Int, nc)
        merge_with = zeros(Int, length(groups))
        keep = fill(true, eachindex(groups))
        needs_merging = false
        for (i, group) in enumerate(groups)
            for g in group
                owner = owned[g]
                if owner == 0
                    owned[g] = i
                else
                    needs_merging = true
                    merge_target = merge_with[i]
                    @assert merge_target == 0 || merge_target == owner "Complicated merging not implemented"
                    merge_with[i] = owner
                    keep[i] = false
                end
            end
        end
        if needs_merging
            new_groups = deepcopy(groups)
            for i in eachindex(groups)
                if !keep[i]
                    merge_index = merge_with[i]
                    target = new_groups[merge_index]
                    nm = length(target)
                    for c in groups[i]
                        push!(target, c)
                    end
                    unique!(target)
                end
            end
            return new_groups[keep]
        else
            return groups
        end
    end
end
