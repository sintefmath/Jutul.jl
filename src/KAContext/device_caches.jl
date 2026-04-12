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

function Jutul.diagonal_view(cache::GenericAutoDiffCache_device)
    dpos = cache.diagonal_positions
    if isnothing(dpos)
        return nothing
    else
        return view(cache.entries, :, dpos)
    end
end

Jutul.ad_dims(c::GenericAutoDiffCache_device) = (number_of_entities(c), equations_per_entity(c), number_of_partials(c))
Jutul.ad_dims(c::CompactAutoDiffCache_device) = (number_of_entities(c), equations_per_entity(c), number_of_partials(c))

Base.eltype(::GenericAutoDiffCache_device{N, E, ∂x}) where {N, E, ∂x} = ∂x

# ────────────────────────────────────────────────────────────────────────────────
# PoissonDeviceCache
# Wraps a GenericAutoDiffCache_device and adds device-side half-face-map arrays
# so that update_equation_for_entity! can run entirely on the device.
# ────────────────────────────────────────────────────────────────────────────────

"""
    PoissonDeviceCache{C, V}

Device-side cache for any `AbstractPoissonEquation`.  It wraps the inner
`GenericAutoDiffCache_device` and adds the half-face-map connectivity arrays
transferred to device memory.

Fields:
- `inner`         – `GenericAutoDiffCache_device` holding entries/jacobian-positions.
- `disc_cells`    – device integer array: neighbour cell for each half-face entry.
- `disc_faces`    – device integer array: face index for each half-face entry.
- `disc_face_pos` – device integer array (length nc+1): CSR-style position array.
- `time_dependent`– whether the equation has an accumulation term.
"""
struct PoissonDeviceCache{C<:GenericAutoDiffCache_device, V} <: JutulAutoDiffCache
    inner::C
    disc_cells::V
    disc_faces::V
    disc_face_pos::V
    time_dependent::Bool
end

# ── Forward all JutulAutoDiffCache accessors to the inner cache ──────────────
Jutul.get_entries(c::PoissonDeviceCache)              = Jutul.get_entries(c.inner)
Jutul.equations_per_entity(c::PoissonDeviceCache)     = Jutul.equations_per_entity(c.inner)
Jutul.number_of_entities(c::PoissonDeviceCache)       = Jutul.number_of_entities(c.inner)
Jutul.number_of_partials(c::PoissonDeviceCache)       = Jutul.number_of_partials(c.inner)
Jutul.ad_dims(c::PoissonDeviceCache)                  = Jutul.ad_dims(c.inner)
Jutul.diagonal_view(c::PoissonDeviceCache)            = Jutul.diagonal_view(c.inner)

Base.eltype(c::PoissonDeviceCache) = Base.eltype(c.inner)
