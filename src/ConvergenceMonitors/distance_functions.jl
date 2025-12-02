using LinearAlgebra
"""
    compute_distance(report; distance_function = r -> scaled_residual_norm(r), mapping = v -> maximum(v))

Compute distance from convergence using a user-defined distance function, and
optionally apply a mapping to the distance. The function returns the distance
and the names of equation residual norms used in the distance computation.
"""
function compute_distance(report;
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
        @assert length(δ) == length(m) "Pooling is only supported if number 
        of distances equals number of residual measures"
        δ, names = pool_distances(δ, names, pools, pooling_norm; pooling_args...)
    end
    δ = δ isa Number ? [δ] : δ

    return δ, names

end

function pool_distances(d, names::Vector{String}, pools::Vector{Vector{Int64}}, p=Inf; check_coverage=true, pool_names=nothing)

    if check_coverage
        pix = unique(vcat(pools...))
        @assert all([k ∈ pix for k in 1:length(d)]) "
        Pools do not cover all residual measures"
    end
    m_pooled = Vector{Float64}(undef, length(pools))
    pools_str = Vector{Vector{String}}(undef, length(pools))
    for (k, pool) in enumerate(pools)
        mp = norm(max.(d[pool].-1, 0.0), p) .+ 1
        m_pooled[k] = mp
        pools_str[k] = names[pool]
    end

    if pool_names !== nothing
        @assert length(pool_names) == length(pools) "Number of pool names must match number of pools"
        pools_str = pool_names
    end

    return m_pooled, pools_str

end

function pool_distances(d, names::Vector{String}, pools::Union{Vector{String}, Vector{Regex}}, args...;
    search_fn=contains, kwargs...)

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
    return pool_distances(d, names, pools_int, args...;  pool_names=pool_names, kwargs...)

end