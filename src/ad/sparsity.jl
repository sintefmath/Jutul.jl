# function unpack_tag(v::Type{ForwardDiff.Dual{T, F, N}}, t::Symbol = :entity) where {T, F, N}
#     @info "???"
#     if v isa Tuple
#         if t == :entity
#             out = T[2]
#         elseif t == :model
#             out = T[1]
#         else
#             out = T
#         end
#     else
#         out = T
#     end
#     return unpack_tag(out)
# end

function unpack_tag(v::Type{ForwardDiff.Dual{T, F, N}}) where {T, F, N}
    return unpack_tag(T)
end

function unpack_tag(v::ForwardDiff.Tag{F, T}) where {F, T}
    if T <: JutulEntity
        return T()
    else
        return nothing
    end
end

unpack_tag(A::AbstractArray, arg...) = unpack_tag(eltype(A), arg...)
unpack_tag(::Any, arg...) = nothing

struct SparsityTracingWrapper{T, N, D} <: AbstractArray{T, N}
    data::Array{D, N}
end

"""
    SparsityTracingWrapper(x::AbstractArray{T, N}) where {T, N}

Create a sparsity tracing wrapper for a numeric array. This wrapped array
produces outputs that have the same value as the wrapped type, but contains a
SparsityTracing seeded value with seed equal to the column index (if matrix) or
linear index (if vector).
"""
function SparsityTracingWrapper(x::AbstractArray{T, N}) where {T, N}
    return SparsityTracingWrapper{Float64, N, T}(x)
end

Base.parent(A::SparsityTracingWrapper) = A.data
Base.size(A::SparsityTracingWrapper) = size(A.data)
Base.axes(A::SparsityTracingWrapper) = axes(A.data)

function Base.getindex(A::SparsityTracingWrapper, i)
    if i isa Colon
        return A
    else
        return map(i -> A[i], i)
    end
end

function Base.getindex(A::SparsityTracingWrapper{T, D, <:Any}, I, J) where {T, D}
    @assert D > 1
    if I isa Colon
        I = axes(A, 1)
    end
    if J isa Colon
        J = axes(A, 2)
    end
    Ts = Jutul.ST.ADval{T}
    n = length(I)
    m = length(J)
    out = Matrix{Ts}(undef, n, m)
    for (i, ix) in enumerate(I)
        for (j, jx) in enumerate(J)
            out[i, j] = A[ix, jx]
        end
    end
    return out
end

function Base.getindex(A::SparsityTracingWrapper, i::Int, j::Int)
    return as_tracer(A.data[i, j], j)
end

function Base.getindex(A::SparsityTracingWrapper{T, 2, D}, ix::Int) where {T, D}
    n, m = size(A)
    zero_ix = ix - 1
    i = (zero_ix ÷ m) + 1
    j = mod(zero_ix, m) + 1
    return as_tracer(A.data[i, j], j)
end

function Base.getindex(A::SparsityTracingWrapper{T, 1, D}, i::Int) where {T, D}
    return as_tracer(A.data[i], i)
end

function as_tracer(x::Real, leaf_init)
    return Jutul.ST.ADval{Float64}(value(x), Jutul.ST.DerivLeaf(leaf_init))
end

function as_tracer(x, leaf_init)
    return x
end

function create_mock_state(state, tag, entities = ad_entities(state); subkeys = nothing)
    no_provided_keys = isnothing(subkeys)
    mock_state = JutulStorage()
    n = entities[tag].n
    for k in keys(state)
        v = state[k]
        tag_matches = unpack_tag(v) == tag
        key_matches = no_provided_keys || (k in subkeys)
        if tag_matches && key_matches
            # Assign mock value with tracer
            new_v = SparsityTracingWrapper(v)
        else
            # Assign mock value as doubles
            new_v = as_value(v)
        end
        mock_state[k] = new_v
    end
    return mock_state
end

function ad_entities(state)
    entity_count(x::AbstractVector) = length(x)
    entity_count(x::AbstractMatrix) = size(x, 2)

    out = Dict()
    for k in keys(state)
        v = state[k]
        tag = unpack_tag(v)
        if !isnothing(tag)
            n = entity_count(v)
            T = eltype(v)
            if haskey(out, tag)
                old = out[tag]
                @assert old.n == n && old.T == T "Inconsistent numbering for $k"
            else
                out[tag] = (n = n, T = T)
            end
        end
    end
    return out
end

function determine_sparsity(F!, n, state, state0, tag, entities, N = entities[tag].n)
    mstate = create_mock_state(state, tag, entities)
    mstate0 = create_mock_state(state0, tag, entities)

    out = ST.create_advec(zeros(n))
    J = [Vector{Int64}() for i in 1:N]
    for i in 1:N
        @inbounds F!(out, mstate, mstate0, i)
        # Take the sum over all return values to reduce to scalar.
        # This should accumulate the full "entity" pattern if some
        # equations have a different stencil.
        V = sum(out)
        D = ST.deriv(V)
        J[i] = D.nzind
    end
    return J
end

function determine_sparsity_simple(F, model, state, state0 = nothing; variant = missing)
    entities = ad_entities(state)
    sparsity = Dict()
    if ismissing(variant)
        subkeys = nothing
    else
        if variant == :parameters
            subkeys = keys(get_parameters(model))
        else
            @assert variant == :variables
            subkeys = keys(get_variables(model))
        end
    end
    for (k, v) in entities
        mstate = create_mock_state(state, k, entities, subkeys = subkeys)
        if isnothing(state0)
            f_ad = F(mstate)
        else
            mstate0 = create_mock_state(state0, k, entities, subkeys = subkeys)
            f_ad = F(mstate, mstate0)
        end
        V = sum(f_ad)
        if V isa AbstractFloat || V isa Integer
            S = zeros(Int64, 0)
        else
            D = ST.deriv(V)
            S = D.nzind
        end
        sparsity[k] = S
    end
    return sparsity
end
