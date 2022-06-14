function generic_cache_declare_pattern(cache::GenericAutoDiffCache)
    pos = cache.vpos
    var = cache.variables
    J = cache.variables
    I = similar(J)
    n = number_of_entities(cache)
    for i in 1:n
        for j in pos[i]:(pos[i+1]-1)
            I[j] = var[j]
        end
    end
    return (I, J)
end

number_of_partials(::Type{ForwardDiff.Dual{T, V, N}}) where {T,V,N} = N

entity(::GenericAutoDiffCache{<:Any, E}) where E = E
equations_per_entity(::GenericAutoDiffCache{N}) where N = N
number_of_entities(c::GenericAutoDiffCache) = length(c.vpos)-1
number_of_partials(c::GenericAutoDiffCache{N, E, ∂T}) where {N, E, ∂T} = number_of_partials(∂T)

