export ConservationLaw
export allocate_array_ad, get_ad_unit_scalar, update_values!
export value, find_sparse_position

function allocate_array_ad(n::R...; context::TervContext = DefaultContext(), diag_pos = nothing, npartials = 1) where {R<:Integer}
    # allocate a n length zero vector with space for derivatives
    T = float_type(context)
    if npartials == 0
        A = allocate_array(context, T(0), n...)
    else
        if isa(diag_pos, AbstractVector)
            @assert n[1] == length(diag_pos) "diag_pos must be specified for all columns."
            d = map(x -> get_ad_unit_scalar(T(0.0), npartials, x), diag_pos)
            A = allocate_array(context, d, 1, n[2:end]...)
        else
            d = get_ad_unit_scalar(T(0.0), npartials, diag_pos)
            A = allocate_array(context, d, n...)
        end
    end
    return A
end

function allocate_array_ad(v::AbstractVector; context = DefaultContext(), diag_pos = nothing, npartials = 1)
    # create a copy of a vector as AD
    v_AD = allocate_array_ad(length(v), context = context, diag_pos = diag_pos, npartials = npartials)
    update_values!(v_AD, v)
end

# Allocators 
function allocate_array_ad(v::AbstractMatrix; context = DefaultContext(), diag_pos = nothing, npartials = 1)
    # create a copy of a vector as AD
    v_AD = allocate_array_ad(size(v)..., context = context, diag_pos = diag_pos, npartials = npartials)
    update_values!(v_AD, v)
end

function get_ad_unit_scalar(v::T, npartials, diag_pos = nothing) where {T<:Real}
    # Get a scalar, with a given number of zero derivatives. A single entry can be specified to be non-zero
    if npartials > 0
        v = ForwardDiff.Dual{T}(v, ntuple(x -> T.(x == diag_pos), npartials))
    end
    return v
end

function update_values!(v::AbstractArray, next::AbstractArray)
    @. v = v - value(v) + next
end

@inline function value(x)
    return ForwardDiff.value(x)
end

function value(d::Dict)
    v = copy(d)
    for key in keys(v)
        v[key] = value.(v[key])
    end
    return v
end

function find_sparse_position(A::SparseMatrixCSC, row, col)
    for pos = A.colptr[col]:A.colptr[col+1]-1
        if A.rowval[pos] == row
            return pos
        end
    end
    return 0
end