"""
Abstract type for domains where equations can be defined
"""
abstract type JutulDomain end

export physical_representation
"""
    physical_representation(x)

Get the physical representation of an object. The physical representation is
usually some kind of mesh or domain that represents a physical domain.
"""
physical_representation(x) = x

export DiscretizedDomain
struct DiscretizedDomain{G, D, E, M} <: JutulDomain
    representation::G
    discretizations::D
    entities::E
    global_map::M
end

"""
    physical_representation(x::DiscretizedDomain)

Get the underlying physical representation (domain or mesh) that was discretized.
"""
physical_representation(x::DiscretizedDomain) = x.representation

function Base.show(io::IO, d::DiscretizedDomain)
    disc = d.discretizations
    p = physical_representation(d)
    if isnothing(disc)
        print(io, "DiscretizedDomain with $p\n")
    else
        print(io, "DiscretizedDomain with $p and discretizations for $(join(keys(d.discretizations), ", "))\n")
    end
end

"""
    DiscretizedDomain(domain, disc = nothing)

A type for a discretized domain of some other domain or mesh. May contain one or
more discretizations as-needed to write equations.
"""
function DiscretizedDomain(domain::JutulDomain, disc = nothing; global_map = TrivialGlobalMap())
    entities = declare_entities(domain)
    u = Dict{JutulEntity, Int}()
    for (entity, num) in entities
        @assert num >= 0 "Units must have non-negative counts. $entity had $num values."
        u[entity] = num
    end
    return DiscretizedDomain(domain, disc, u, global_map)
end

export DataDomain
struct DataDomain{R, E, D} <: JutulDomain
    representation::R
    entities::E
    data::D
end

function hasentity(d::Union{DataDomain, DiscretizedDomain}, e)
    return haskey(d.entities, e)
end

=======
function declare_entities(domain::DataDomain)
    return ((entity = key, count = val) for (key, val) in domain.entities)
end

"""
    physical_representation(x::DataDomain)

Get the underlying physical representation (domain or mesh) that is wrapped.
"""
physical_representation(x::DataDomain) = x.representation

function Base.show(io::IO, t::MIME"text/plain", d::DataDomain)
    # disc = d.discretizations
    p = physical_representation(d)
    print(io, "DataDomain wrapping $p")
    data = d.data
    k = keys(data)
    n = length(k)
    if n == 0
        print(io, " with no additional data.\n")
    else
        print(io, " with $n data fields added:\n")
        unique_entities = unique(map(last, values(data)))
        for u in unique_entities
            if u == NoEntity()
                s = "(no associated entity)"
            else
                ne = count_entities(d, u)
                s = "$ne $(typeof(u))"
            end
            print(io, "  $s\n")
            for (k, v) in data
                vals, e = v
                if e == u
                    sz = join(map(x -> "$x", size(vals)), "×")
                    print(io, "    :$k => $sz $(typeof(vals))\n")
                end
            end
        end
    end

end

"""
    DataDomain(domain::JutulDomain; property1 = p1, property2 = p2, ...)

A wrapper around a domain that allows for storing of entity-associated data.

Example:
```julia
# Grid with 6 cells and 7 interior faces
g = CartesianMesh((2, 3))
d = DataDomain(g)
d[:cell_vec] = rand(6) #ok, same as:
d[:cell_vec, Cells()] = rand(6) #ok
d[:cell_vec, Faces()] = rand(6) #not ok!
d[:face_vec, Faces()] = rand(7) #ok!
# Can also add general arrays if last dimension == entity dimension
d[:cell_vec, Cells()] = rand(10, 3, 6) #ok
# Can add general data too, but needs to be specified
d[:not_on_face_or_cell, nothing] = rand(3) # also ok
```
"""
function DataDomain(domain::JutulDomain; kwarg...)
    entities = declare_entities(domain)
    u = Dict{JutulEntity, Int}()
    for (entity, num) in entities
        @assert num >= 0 "Units must have non-negative counts. $entity had $num values."
        u[entity] = num
    end
    u[NoEntity()] = 1
    data = OrderedDict{Symbol, Any}()
    Ω = DataDomain(domain, u, data)
    for (k, v) in kwarg
        if v isa Tuple{Any, JutulEntity}
            Ω[k, last(v)] = first(v)
        else
            Ω[k] = v
        end
    end
    add_default_domain_data!(Ω, domain)
    return Ω
end

add_default_domain_data!(Ω::DataDomain, domain) = nothing

function Base.setindex!(domain::DataDomain, val, key::Symbol, entity = Cells())
    if ismissing(entity) || isnothing(entity)
        entity = NoEntity()
    end
    n = count_entities(domain, entity)

    function validate(val::AbstractVector)
        d = length(val)
        @assert d == n "Number of values for Vector $key defined on $entity should be $n, was $d"
        return val
    end
    function validate(val::AbstractMatrix)
        d = size(val, 2)
        @assert d == n "Number of columns for Matrix $key defined on $entity should be $n, was $d"
        return val
    end
    function validate(val::AbstractArray)
        d = last(size(val))
        @assert d == n "Last index of multidimensional array $key defined on $entity should have size $n, was $d"
        return val
    end
    function validate(val)
        # Repeat over entire domain
        return [copy(val) for i in 1:n]
    end

    if entity != NoEntity()
        # If data is associated with entity it must be validated
        val = validate(val)
    end
    domain.data[key] = (val, entity)
    return val
end

function Base.getindex(domain::DataDomain, key::Symbol, entity = nothing)
    v, e = domain.data[key]
    if !isnothing(entity)
        @assert e == entity "Expected property $key to be defined for $entity, but was stored as $e"
    end
    return v
end

function Base.keys(domain::DataDomain)
    return keys(domain.data)
end

function Base.haskey(domain::DataDomain, name::Symbol)
    return haskey(domain.data, name)
end


function Base.haskey(domain::DataDomain, name::Symbol, entity)
    if ismissing(entity) || isnothing(entity)
        entity = NoEntity()
    end
    return haskey(domain.data, name) && last(domain.data[name]) == entity
end

Base.iterate(domain::DataDomain) = Base.iterate(domain.data)
Base.pairs(domain::DataDomain) = Base.pairs(domain.data)
