import Base: getindex, @propagate_inbounds, parent, size, axes
using ForwardDiff
import ForwardDiff: value, Dual

struct LocalPerspectiveAD{T, N, A<:AbstractArray{T,N}, I} <: AbstractArray{T,N}
    index::I
    data::A
end

function LocalPerspectiveAD(a::A, index::I_t) where {A<:AbstractArray, I_t<:Integer}
    LocalPerspectiveAD{eltype(a), ndims(A), A, I_t}(index, a)
end

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
