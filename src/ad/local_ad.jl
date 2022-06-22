import Base: getindex, @propagate_inbounds, parent, size, axes
using ForwardDiff

struct LocalPerspectiveAD{T, N, A<:AbstractArray{T,N}, I} <: AbstractArray{T,N}
    index::I
    data::A
end

function LocalPerspectiveAD(a::A, index::I_t) where {A<:AbstractArray, I_t<:Integer}
    LocalPerspectiveAD{eltype(a), ndims(A), A, I_t}(index, a)
end

struct LocalStateAD{T, I, E} # Data type, index, entity tag
    index::I
    data::T
end

struct ValueStateAD{T} # Data type
    data::T
end

as_value(x::Union{NamedTuple,AbstractDict,JutulStorage}) = ValueStateAD(x)

export local_ad
local_ad(v::AbstractArray, i::Integer) = LocalPerspectiveAD(v, i)
local_ad(v::ConstantWrapper, i::Integer) = v
local_ad(v, ::Nothing) = as_value(v)
local_ad(v, i) = v


@inline local_entity(a::LocalPerspectiveAD) = a.index

@inline function value_or_ad(A::LocalPerspectiveAD{T}, v::T, entity) where T
    if entity == local_entity(A)
        return v
    else
        return T(value(v))
    end
end

@propagate_inbounds Base.getindex(A::LocalPerspectiveAD{T}, i::Int) where T = value_or_ad(A, A.data[i], i)
@propagate_inbounds Base.getindex(A::LocalPerspectiveAD{T}, i::Int, j::Int) where T = value_or_ad(A, A.data[i, j], j)


Base.parent(A::LocalPerspectiveAD) = A.data
Base.size(A::LocalPerspectiveAD) = size(A.data)
Base.axes(A::LocalPerspectiveAD) = axes(A.data)
parenttype(::Type{LocalPerspectiveAD{T,N,A,I}}) where {T,N,A,I} = A

function Base.getproperty(state::LocalStateAD{T, I, E}, f::Symbol) where {T, I, E}
    myfn(x::AbstractArray{T}, ::Type{T}, index) where T = local_ad(x, index)
    myfn(x, t, index) = as_value(x)
    myfn(x::ConstantWrapper, t, index) = x

    index = getfield(state, :index)
    inner_state = getfield(state, :data)
    val = getproperty(inner_state, f)
    return myfn(val, E, index)
end

function Base.getproperty(state::ValueStateAD{T}, f::Symbol) where {T}
    myfn(x, t, index) = as_value(x)
    myfn(x::ConstantWrapper, t, index) = x

    index = getfield(state, :index)
    inner_state = getfield(state, :data)
    val = getproperty(inner_state, f)
    return myfn(val, E, index)
end

function local_ad(x::T, index::I, ad_tag::∂T) where {T, I<:Integer, ∂T}
    return LocalStateAD{T, I, ad_tag}(index, x)
end

function Base.show(io::IO, t::MIME"text/plain", x::LocalStateAD{T, I, E}) where {T, I, E}
    print(io, "Local state for $E -> $I with fields $(keys(getfield(x, :data)))")
end
