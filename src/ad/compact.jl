"""
Get entries of autodiff cache. Entries are AD vectors that hold values and derivatives.
"""
@inline function get_entries(c::CompactAutoDiffCache)
    return c.entries
end

@inline function get_entry(c::CompactAutoDiffCache{I, D}, index, eqNo)::D where {I, D}
    return @inbounds get_entries(c)[eqNo, index]
end

@inline function get_value(c::CompactAutoDiffCache, arg...)
    return value(get_entry(c, arg...))
end

@inline function get_jacobian_pos(c::CompactAutoDiffCache{I}, index, eqNo, partial_index, pos) where {I}
    return @inbounds pos[(eqNo - 1) * c.npartials + partial_index, index]
end

@inline function get_jacobian_pos(c::CompactAutoDiffCache{I}, index, eqNo, partial_index) where {I}
    return @inbounds c.jacobian_positions[(eqNo - 1) * c.npartials + partial_index, index]
end

entity(c::CompactAutoDiffCache) = c.entity

# entity(::GenericAutoDiffCache{<:Any, E}) where E = E
equations_per_entity(c::CompactAutoDiffCache) = c.equations_per_entity
number_of_entities(c::CompactAutoDiffCache) = c.number_of_entities
number_of_partials(c::CompactAutoDiffCache{I, D}) where {I, D} = number_of_partials(D)
