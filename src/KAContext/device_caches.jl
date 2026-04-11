# Device wrappers for AD caches that hold device arrays but
# keep the same interface as their CPU counterparts.

using ..Jutul: JutulEntity

"""
Device wrapper for CompactAutoDiffCache – holds device arrays.
"""
struct CompactAutoDiffCache_device{I, ∂x, E, P} <: JutulAutoDiffCache
    entries::E
    entity
    jacobian_positions::P
    equations_per_entity::I
    number_of_entities::I
    npartials::I
end

function CompactAutoDiffCache_device(cpu_cache::CompactAutoDiffCache, entries_dev, jpos_dev)
    return CompactAutoDiffCache_device{
        typeof(cpu_cache.equations_per_entity),
        eltype(entries_dev),
        typeof(entries_dev),
        typeof(jpos_dev)
    }(
        entries_dev,
        cpu_cache.entity,
        jpos_dev,
        cpu_cache.equations_per_entity,
        cpu_cache.number_of_entities,
        cpu_cache.npartials
    )
end

Jutul.get_entries(c::CompactAutoDiffCache_device) = c.entries
Jutul.equations_per_entity(c::CompactAutoDiffCache_device) = c.equations_per_entity
Jutul.number_of_entities(c::CompactAutoDiffCache_device) = c.number_of_entities
Jutul.number_of_partials(c::CompactAutoDiffCache_device) = c.npartials

"""
Device wrapper for GenericAutoDiffCache.
"""
struct GenericAutoDiffCache_device{N, E, ∂x, A, P, M, D} <: JutulAutoDiffCache
    entries::A
    vpos::P
    variables::P
    jacobian_positions::M
    diagonal_positions::D
    number_of_entities_target::Int
    number_of_entities_source::Int
end

function GenericAutoDiffCache_device(cpu_cache::GenericAutoDiffCache{N, E, ∂x}, entries_dev, jpos_dev, vpos_dev, vars_dev, dpos_dev) where {N, E, ∂x}
    return GenericAutoDiffCache_device{N, E, ∂x,
        typeof(entries_dev), typeof(vpos_dev), typeof(jpos_dev), typeof(dpos_dev)
    }(
        entries_dev,
        vpos_dev,
        vars_dev,
        jpos_dev,
        dpos_dev,
        cpu_cache.number_of_entities_target,
        cpu_cache.number_of_entities_source
    )
end

Jutul.get_entries(c::GenericAutoDiffCache_device) = c.entries
function Jutul.equations_per_entity(::GenericAutoDiffCache_device{N}) where N
    return N
end
function Jutul.number_of_entities(c::GenericAutoDiffCache_device)
    return length(c.vpos) - 1
end
function Jutul.number_of_partials(::GenericAutoDiffCache_device{N, E, ∂x}) where {N, E, ∂x}
    return Jutul.number_of_partials(∂x)
end
