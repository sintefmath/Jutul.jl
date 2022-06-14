function generic_cache_declare_pattern(cache::GenericAutoDiffCache)
    J = cache.variables
    I = similar(J)
    n = number_of_entities(cache)
    for i in 1:n
        for j in vrange(cache, i)
            I[j] = i
        end
    end
    return (I, J)
end

number_of_partials(::Type{ForwardDiff.Dual{T, V, N}}) where {T,V,N} = N

entity(::GenericAutoDiffCache{<:Any, E}) where E = E
equations_per_entity(::GenericAutoDiffCache{N}) where N = N
number_of_entities(c::GenericAutoDiffCache) = length(c.vpos)-1
number_of_partials(c::GenericAutoDiffCache{N, E, ∂T}) where {N, E, ∂T} = number_of_partials(∂T)

vrange(c::GenericAutoDiffCache, i) = c.vpos[i]:(c.vpos[i+1]-1)
get_entries(c::GenericAutoDiffCache) = c.entries

@inline function get_jacobian_pos(c::GenericAutoDiffCache, index, eqNo, partial_index, pos)
    np = number_of_partials(c)
    @inbounds pos[(eqNo-1)*np + partial_index, index]
end

diagonal_view(cache::GenericAutoDiffCache) = view(cache.entries, :, cache.diagonal_positions)
