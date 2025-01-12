import Base: getindex, @propagate_inbounds, parent, size, axes

struct LocalPerspectiveAD{T, N, A<:AbstractArray{T,N}, I} <: AbstractArray{T,N}
    index::I
    data::A
end

function LocalPerspectiveAD(a::A, index::I_t) where {A<:AbstractArray, I_t<:Integer}
    return LocalPerspectiveAD{eltype(a), ndims(A), A, I_t}(index, a)
end

struct LocalStateAD{T, I, E} # Data type, index, entity tag
    index::I
    data::T
end

Base.keys(x::LocalStateAD) = keys(getfield(x, :data))

struct ValueStateAD{T} # Data type
    data::T
end

Base.keys(x::ValueStateAD) = keys(getfield(x, :data))

function convert_to_immutable_storage(x::ValueStateAD)
    data = getfield(x, :data)
    data = convert_to_immutable_storage(data)
    return ValueStateAD(data)
end

const StateType = Union{NamedTuple,AbstractDict,JutulStorage}

as_value(x::StateType) = ValueStateAD(x)

export local_ad
@inline local_ad(v::AbstractArray, i::Int) = LocalPerspectiveAD(v, i)
@inline local_ad(v, ::Nothing) = as_value(v)
@inline local_ad(v, i) = v


@inline function new_entity_index(state::LocalStateAD{T, I, E}, index::I) where {T, I, E}
    return LocalStateAD{T, I, E}(index, getfield(state, :data))
end

@inline function new_entity_index(x, index)
    return x
end


@inline local_entity(a::LocalPerspectiveAD) = a.index

@inline function value_or_ad(A::LocalPerspectiveAD{T}, v::T, entity) where T
    if entity === local_entity(A)
        return v
    else
        return T(value(v))
    end
end

@inline @propagate_inbounds function Base.getindex(A::LocalPerspectiveAD{T, N, V, I}, i::Int) where  {T, N, V<:AbstractVector, I}
    d = A.data[i]
    return value_or_ad(A, d, i)
end

@inline @propagate_inbounds function Base.getindex(A::LocalPerspectiveAD{T, N, V, I}, linear_ix::Int) where {T, N, V<:AbstractMatrix, I}
    # Linear indexing into matrix
    data = A.data
    n = size(data, 1)
    i = ((linear_ix - 1) % n) + 1
    j = div(linear_ix-1, n) + 1
    return A[i, j]
end

@inline @propagate_inbounds function Base.getindex(A::LocalPerspectiveAD{T}, i::Int, j::Int) where T
    d = A.data[i, j]
    return value_or_ad(A, d, j)
end

@inline Base.parent(A::LocalPerspectiveAD) = A.data
@inline Base.size(A::LocalPerspectiveAD) = size(A.data)
@inline Base.axes(A::LocalPerspectiveAD) = axes(A.data)
@inline parenttype(::Type{LocalPerspectiveAD{T,N,A,I}}) where {T,N,A,I} = A
@inline Base.haskey(state::LocalStateAD, f::Symbol) = haskey(getfield(state, :data), f)
@inline function Base.haskey(state::LocalStateAD{NamedTuple{keys, vals}, I, E}, f::Symbol) where {keys, vals, I, E}
    return f in keys
end
@inline Base.haskey(state::ValueStateAD, f::Symbol) = haskey(getfield(state, :data), f)


# Match in type - pass index on
@inline next_level_local_ad(x::AbstractArray{T}, ::Type{T}, index) where T = local_ad(x, index)

@inline function next_level_local_ad(x::AbstractArray{E}, ::Type{T}, index) where {T, E}
    # Mismatch in AD type - take value
    if numerical_type(E) == T
        return local_ad(x, index)
    else
        return as_value(x)
    end
end

@inline function next_level_local_ad(x, ::Type{T}, index) where T
    if numerical_type(x) == T
        return local_ad(x, index)
    else
        # Mismatch in AD type - take value
        return as_value(x)
    end
end

"""
    numerical_eltype(x::AbstractArray{T}) where T

Get the numerical eltype (i.e. the inner type of the element type that could
potentially be AD)
"""
@inline function numerical_eltype(x::AbstractArray{T}) where T
    return numerical_type(T)
end

@inline function numerical_type(::Type{T}) where T
    return T
end

"""
    numerical_type(::T) where T

Get the numerical eltype (i.e. the inner type of the element type that could
potentially be AD). This function should be overloaded if you have a custom
type that wraps a numeric/potentially AD type.
"""
@inline function numerical_type(::T) where T<:Real
    # Default numerical type is just the type itself.
    return T
end

# Nested states
@inline function next_level_local_ad(x::StateType, E::Type, index)
    local_ad(x, index, E)
end

@inline function Base.getproperty(state::LocalStateAD{T, I, E}, f::Symbol) where {T, I, E}
    index = getfield(state, :index)
    inner_state = getfield(state, :data)
    val = getproperty(inner_state, f)
    return next_level_local_ad(val, E, index)
end

@inline function Base.getindex(state::LocalStateAD, s::Symbol)
    Base.getproperty(state, s)
end

@inline function Base.getproperty(state::ValueStateAD{T}, f::Symbol) where {T}
    inner_state = getfield(state, :data)
    val = getproperty(inner_state, f)
    return as_value(val)
end

@inline function Base.getindex(state::ValueStateAD, f::Symbol)
    return getproperty(state, f)
end

"""
    local_ad(state::T, index::I, ad_tag::∂T) where {T, I<:Integer, ∂T}

Create local_ad for state for index I of AD tag of type ad_tag
"""
@inline function local_ad(state, index, ad_tag)
    local_state_ad(state, index, ad_tag)
end

@inline function local_state_ad(state::T, index::I, ad_tag::∂T) where {T, I<:Integer, ∂T}
    return LocalStateAD{T, I, ad_tag}(index, state)
end

function Base.show(io::IO, t::MIME"text/plain", x::LocalStateAD{T, I, E}) where {T, I, E}
    print(io, "Local state for $(unpack_tag(E)) -> $(getfield(x, :index)) with fields $(keys(getfield(x, :data)))")
end
