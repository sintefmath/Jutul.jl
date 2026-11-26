"""
    δ, names = compute_distance(report; distance_function = v -> max.(v, 1.0), pools = nothing, pooling_norm = Inf, pooling_args = NamedTuple())

Compute the distance from convergence for a nonlinear iteration report (the
`:errors` entry of a step report). The residual measures in the report, scaled
by their tolerances, are flattened into a vector and passed to
`distance_function`. A distance of one or less means that the corresponding
measure has converged.

If `pools` is given, the distances are pooled into one distance per pool using
the `pooling_norm`-norm of the excess distance (see `pool_distances`). Pools can
be given as a vector of index vectors, a vector of strings or regexes matched
against the measure names, or `:all` to pool all measures into one. Additional
keyword arguments to `pool_distances` can be passed through `pooling_args`.

Returns the distances and the names of the residual measures or pools.
"""
function compute_distance(
        report;
        distance_function = v -> max.(v, 1.0),
        pools = nothing,
        pooling_norm = Inf,
        pooling_args = NamedTuple(),
    )

    # Compute scaled residual measures
    measures = residual_measures(report)
    m, names = flatten_dict(measures)
    δ = distance_function(m)
    if !isnothing(pools)
        @assert length(δ) == length(m) "Pooling is only supported if number of distances equals number of residual measures"
        δ, names = pool_distances(δ, names, pools, pooling_norm; pooling_args...)
    end
    δ = δ isa Number ? [δ] : δ

    return δ, names

end

"""
    δ_pooled, pool_names = pool_distances(d, names, pools, p = Inf; check_coverage = true, pool_names = nothing)

Pool the distances `d` into one distance per pool. The pooled distance is
`norm(max.(d[pool] .- 1, 0), p) + 1`, so that a pool has converged (distance
one) if and only if all its members have converged. Pools can be given as a
vector of index vectors, a vector of strings or regexes matched against `names`
(using the keyword argument `search_fn`, defaulting to `contains`), or `:all`.
If `check_coverage` is true, the pools must cover all measures. The pools are
named by `pool_names` if given, otherwise by the names of their members.
"""
function pool_distances(d, names::Vector{String}, pools::Vector{Vector{Int64}}, p = Inf; check_coverage = true, pool_names = nothing)

    if check_coverage
        pix = unique(vcat(pools...))
        @assert all(in(pix), 1:length(d)) "Pools do not cover all residual measures"
    end
    m_pooled = Vector{Float64}(undef, length(pools))
    pools_str = Vector{Vector{String}}(undef, length(pools))
    for (k, pool) in enumerate(pools)
        @views m_pooled[k] = norm(max.(d[pool] .- 1, 0.0), p) + 1
        pools_str[k] = names[pool]
    end

    if pool_names !== nothing
        @assert length(pool_names) == length(pools) "Number of pool names must match number of pools"
        pools_str = pool_names
    end

    return m_pooled, pools_str

end

function pool_distances(
        d, names::Vector{String}, pools::Union{Vector{String}, Vector{Regex}}, args...;
        search_fn = contains, kwargs...
    )

    pools_int = Vector{Vector{Int64}}(undef, length(pools))
    for (k, pool) in enumerate(pools)
        ix = findall(search_fn.(names, pool))
        @assert !isempty(ix) "No residual measures found for pool: $pool"
        pools_int[k] = ix
    end

    return pool_distances(d, names, pools_int, args...; kwargs...)

end

function pool_distances(d, names::Vector{String}, pools::Symbol, args...; kwargs...)

    pools == :all || error("This version only supports :all, got: $pools")
    pools_int = [collect(1:length(names))]
    pool_names = ["AllMeasures"]
    return pool_distances(d, names, pools_int, args...; pool_names = pool_names, kwargs...)

end
