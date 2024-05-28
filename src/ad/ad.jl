export CompactAutoDiffCache, as_value, JutulAutoDiffCache, number_of_entities, get_entries, fill_equation_entries!, matrix_layout

"""
Get number of entities a cache is defined on.
"""
@inline number_of_entities(c::JutulAutoDiffCache) = c.number_of_entities


"""
Number of entities for vector stored in state (just the number of elements)
"""
@inline number_of_entities(c::T) where T<:AbstractVector = length(c)

"""
Number of entities for matrix stored in state (convention is number of columns)
"""
@inline number_of_entities(c::T) where T<:AbstractArray = size(c, 2)

"""
Get the entries of the main autodiff cache for an equation.

Note: This only gets the .equation field's entries.
"""
@inline function get_entries(e::JutulEquation)
    return get_entries(e.equation)
end

@inline function get_entry(c::JutulAutoDiffCache, index, eqNo)
    @inbounds get_entries(c)[eqNo, index]
end

@inline Base.@propagate_inbounds get_entry_impl(x::ForwardDiff.Dual, derNo::Int) = x.partials[derNo]
@inline Base.@propagate_inbounds get_entry_impl(x::Real, derNo::Int) = x

@inline function get_entry(c::JutulAutoDiffCache, index, eqNo, derNo)
    @inbounds get_entry_impl(get_entries(c)[eqNo, index], derNo)
end

@inline function get_entry_val(c::JutulAutoDiffCache, index, eqNo)
    @inbounds value(get_entries(c)[eqNo, index])
end

include("compact.jl")
include("generic.jl")

@inline function set_jacobian_pos!(c::JutulAutoDiffCache, index, eqNo, partial_index, pos)
    set_jacobian_pos!(c.jacobian_positions, index, eqNo, partial_index, c.npartials, pos)
    # c.jacobian_positions[(eqNo-1)*c.npartials + partial_index, index] = pos
end

@inline function get_jacobian_pos(np::I, index, eqNo, partial_index, pos)::I where {I<:Integer}
    @inbounds pos[(eqNo-1)*np + partial_index, index]
end

@inline function set_jacobian_pos!(jpos, index, eqNo, partial_index, npartials, pos)
    jpos[jacobian_cart_ix(index, eqNo, partial_index, npartials)] = pos
end


@inline jacobian_row_ix(eqNo, partial_index, npartials) = (eqNo-1)*npartials + partial_index
@inline jacobian_cart_ix(index, eqNo, partial_index, npartials) = CartesianIndex((eqNo-1)*npartials + partial_index, index)

@inline function ad_dims(cache)
    return (number_of_entities(cache), equations_per_entity(cache), number_of_partials(cache))::NTuple
end

@inline function update_jacobian_entry!(nzval, c::JutulAutoDiffCache, index, eqNo, partial_index, 
                                                                        new_value,
                                                                        pos = c.jacobian_positions)
    ix = get_jacobian_pos(c, index, eqNo, partial_index, pos)
    update_jacobian_inner!(nzval, ix, new_value)
end

@inline function update_jacobian_inner!(nzval, pos, val)
    @inbounds nzval[pos] = val
end

insert_residual_value(::Nothing, ix, v) = nothing
Base.@propagate_inbounds insert_residual_value(r, ix, v) = r[ix] = v

insert_residual_value(::Nothing, ix, e, v) = nothing
Base.@propagate_inbounds insert_residual_value(r, ix, e, v) = r[e, ix] = v

function fill_equation_entries!(nz, r, model, cache::JutulAutoDiffCache)
    nu, ne, np = ad_dims(cache)
    tb = minbatch(model.context, nu)
    @batch minbatch = tb for i in 1:nu
        @inbounds for e in 1:ne
            a = get_entry(cache, i, e)
            insert_residual_value(r, i + nu*(e-1), a.value)
            for d = 1:np
                update_jacobian_entry!(nz, cache, i, e, d, a.partials[d])
            end
        end
    end
end

function diagonal_alignment!(cache, arg...; eq_index = 1:cache.number_of_entities, kwarg...)
    injective_alignment!(cache, arg...; target_index = eq_index, source_index = eq_index, kwarg...)
end

function injective_alignment!(cache::JutulAutoDiffCache, eq, jac, _entity, context;
            pos = nothing,
            row_layout = matrix_layout(context),
            col_layout = row_layout,
            target_index = 1:cache.number_of_entities,
            source_index = 1:cache.number_of_entities,
            number_of_entities_source = nothing,
            number_of_entities_target = nothing,
            number_of_equations_for_entity = missing,
            dims = ad_dims(cache),
            row_offset = 0,
            column_offset = 0,
            target_offset = 0,
            source_offset = 0
    )
    _entity::JutulEntity
    c_entity = entity(cache)
    c_entity::JutulEntity
    if isnothing(pos)
        pos = cache.jacobian_positions
    end
    if _entity == c_entity
        nu_c, ne, np = dims
        if isnothing(number_of_entities_source)
            nu_s = nu_c
        else
            nu_s = number_of_entities_source
        end
        if isnothing(number_of_entities_target)
            nu_t = length(target_index)
        else
            nu_t = number_of_entities_target
        end
        N = length(source_index)
        @assert length(target_index) == N
        if ismissing(number_of_equations_for_entity)
            number_of_equations_for_entity = ne
        end
        do_injective_alignment!(pos, cache, jac, target_index, source_index, nu_t, nu_s, ne, np, row_offset, column_offset, target_offset, source_offset, context, row_layout, col_layout, number_of_equations_for_entity = number_of_equations_for_entity)
    end
end

function do_injective_alignment!(jpos, cache, jac, target_index, source_index, nu_t, nu_s, ne, np, row_offset, column_offset, target_offset, source_offset, context, row_layout, col_layout; number_of_equations_for_entity = ne)
    for index in 1:length(source_index)
        target = target_index[index]
        source = source_index[index]
        for e in 1:ne
            for d = 1:np
                jpos[jacobian_cart_ix(index, e, d, np)] = find_jac_position(
                    jac,
                    target, source,
                    row_offset, column_offset,
                    target_offset, source_offset,
                    e, d,
                    nu_t, nu_s,
                    ne, np,
                    row_layout, col_layout,
                    number_of_equations_for_entity = number_of_equations_for_entity
                )
            end
        end
    end
end



# function do_injective_alignment!(jpos, cache, jac, target_index, source_index, nu_t, nu_s, ne, np, target_offset, source_offset, context::SingleCUDAContext, layout)
#     t = index_type(context)

#     ns = t(length(source_index))
#     nu_t = t(nu_t)
#     dims = (ns, ne, np)
#     target_index = UnitRange{t}(target_index)
#     source_index = UnitRange{t}(source_index)
#     @kernel function cu_injective_align(jpos, 
#                                     @Const(rows), @Const(cols),
#                                     @Const(target_index), @Const(source_index),
#                                     nu_t, nu_s, ne, np, 
#                                     target_offset, source_offset,
#                                     layout)
#         index, e, d = @index(Global, NTuple)
#         target = target_index[index]
#         source = source_index[index]

#         row, col = row_col_sparse(target + target_offset, source + source_offset, e, d, 
#         nu_t, nu_s,
#         ne, np,
#         layout)
        
#         # ix = find_sparse_position_CSC(rows, cols, row, col)

#         T = eltype(cols)
#         ix = 0
#         for pos = cols[col]:cols[col+1]-1
#             if rows[pos] == row
#                 ix = pos
#                 break
#             end
#         end
#         j_ix = jacobian_row_ix(e, d, np)
#         jpos[j_ix, index] = ix
#     end
#     kernel = cu_injective_align(context.device, context.block_size)
    
#     rows = jac.rowVal
#     cols = jac.colPtr
#     event_jac = kernel(jpos, rows, cols, target_index, source_index, nu_t, nu_s, ne, np, t(target_offset), t(source_offset), layout, ndrange = dims)
#     wait(event_jac)
# end

"""
Convert a state containing variables as arrays of doubles
to a state where those arrays contain the same value as Dual types.
The dual type is currently taken from ForwardDiff.
"""
function convert_state_ad(model, state, tag = nothing)
    context = model.context
    stateAD = deepcopy(state)

    primary = get_primary_variables(model)
    # Loop over primary variables and set them to AD, with ones at the correct diagonal
    # @debug "Found $n_partials primary variables."
    last_entity = nothing
    offset = 0
    outstr = ""
    if isnothing(tag)
        outstr *= "Setting up primary variables...\n"
    else
        outstr *= "$tag: Setting up primary variables...\n"
    end
    # Bookkeeping for debug output
    total_number_of_partials = 0
    total_number_of_groups = 0
    for (pkey, pvar) in primary
        u = associated_entity(pvar)
        # Number of partials for this entity
        n_partials = degrees_of_freedom_per_entity(model, u)
        if last_entity != u
            n_entities = count_entities(model.domain, u)
            outstr *= "Variable group:\n\t$(n_entities) $(typeof(u)) with $n_partials partial derivatives each ($(n_partials*n_entities) total).\n"
            # Note: We assume that the variables are sorted by entities.
            # This is asserted for in the model constructor.
            last_entity = u
            # Reset the offset to zero, since we have entered a new group.
            offset = 0
            total_number_of_groups += 1
        end
        # Number of partials this primary variable contributes
        n_local = degrees_of_freedom_per_entity(model, pvar)
        t = get_entity_tag(tag, u)
        outstr *= "→ $pkey:\n\t$n_local of $n_partials partials on all $(typeof(u)), covers $(offset+1) → $(offset + n_local)\n"
        stateAD = initialize_primary_variable_ad!(stateAD, model, pvar, pkey, n_partials, tag = t, offset = offset, context = context)
        offset += n_local
        total_number_of_partials += n_local
    end
    outstr *= "Primary variables set up: $(number_of_degrees_of_freedom(model)) degrees of freedom\n\t → $total_number_of_partials distinct primary variables over $total_number_of_groups different entities.\n"
    secondary = get_secondary_variables(model)
    # Loop over secondary variables and initialize as AD with zero partials
    outstr *= "Setting up secondary variables...\n"
    for (skey, svar) in secondary
        u = associated_entity(svar)
        outstr *= "\t$skey: Defined on $(typeof(u))\n"

        t = get_entity_tag(tag, u)
        n_partials = degrees_of_freedom_per_entity(model, u)
        stateAD = initialize_secondary_variable_ad!(stateAD, model, svar, skey, n_partials, tag = t, context = context)
    end
    @debug outstr
    return stateAD
end

"""
    allocate_array_ad(n[, m]; <keyword arguments>)

Allocate vector or matrix as AD with optionally provided context and a specified non-zero on the diagonal.

# Arguments
- `n::Integer`: number of entries in vector, or number of rows if `m` is given.
- `m::Integer`: number of rows (optional)

# Keyword arguments
- `npartials = 1`: Number of partials derivatives to allocate for each element
- `diag_pos = nothing`: Indices of where to put entities on the diagonal (if any)

Other keyword arguments are passed onto `get_ad_entity_scalar`.

# Examples:

Allocate a vector with a single partial:
```julia-repl
julia> allocate_array_ad(2)
2-element Vector{ForwardDiff.Dual{nothing, Float64, 1}}:
 Dual{nothing}(0.0,0.0)
 Dual{nothing}(0.0,0.0)
```
Allocate a vector with two partials, and set the first to one:
```julia-repl
julia> allocate_array_ad(2, diag_pos = 1, npartials = 2)
2-element Vector{ForwardDiff.Dual{nothing, Float64, 2}}:
 Dual{nothing}(0.0,1.0,0.0)
 Dual{nothing}(0.0,1.0,0.0)
```
Set up a matrix with two partials, where the first column has partials [1, 0] and the second [0, 1]:
```julia-repl
julia> allocate_array_ad(2, 2, diag_pos = [1, 2], npartials = 2)
2×2 Matrix{ForwardDiff.Dual{nothing, Float64, 2}}:
 Dual{nothing}(0.0,1.0,0.0)  Dual{nothing}(0.0,1.0,0.0)
 Dual{nothing}(0.0,0.0,1.0)  Dual{nothing}(0.0,0.0,1.0)
```
"""
function allocate_array_ad(n::R...; context::JutulContext = DefaultContext(), diag_pos = nothing, npartials = 1, kwarg...) where {R<:Integer}
    # allocate a n length zero vector with space for derivatives
    T = float_type(context)
    z_val = zero(T)
    if npartials == 0
        A = allocate_array(context, [z_val], n...)
    else
        if isa(diag_pos, AbstractVector)
            @assert n[1] == length(diag_pos) "diag_pos must be specified for all columns."
            d = map(x -> get_ad_entity_scalar(z_val, npartials, x; kwarg...), diag_pos)
            A = allocate_array(context, d, 1, n[2:end]...)
        else
            d = get_ad_entity_scalar(z_val, npartials, diag_pos; kwarg...)
            A = allocate_array(context, d, n...)
        end
    end
    return A
end

"""
    allocate_array_ad(v::AbstractVector, ...)
Convert vector to AD vector.
"""
function allocate_array_ad(v::AbstractVector; kwarg...)
    # create a copy of a vector as AD
    v_AD = allocate_array_ad(length(v); kwarg...)
    update_values!(v_AD, v)
end

"""
    allocate_array_ad(v::AbstractMatrix, ...)
Convert matrix to AD matrix.
"""
function allocate_array_ad(v::AbstractMatrix; kwarg...)
    # create a copy of a vector as AD
    v_AD = allocate_array_ad(size(v)...; kwarg...)
    update_values!(v_AD, v)
end

"""
    get_ad_entity_scalar(v::Real, npartials, diag_pos = nothing; <keyword_arguments>)

Get scalar with partial derivatives as AD instance.

# Arguments
- `v::Real`: Value of AD variable.
- `npartials`: Number of partial derivatives each AD instance holds.
- `diag_pos` = nothing: Position(s) of where to set 1 as the partial derivative instead of zero.

# Keyword arguments
- `tag = nothing`: Tag for AD instance. Two AD values of the different tag cannot interoperate to avoid perturbation confusion (see ForwardDiff documentation).
"""
function get_ad_entity_scalar(v::T, npartials, diag_pos = nothing; diag_value = 1.0, tag = nothing) where {T<:Real}
    # Get a scalar, with a given number of zero derivatives. A single entry can be specified to be non-zero
    if npartials > 0
        D = diag_value.*ntuple(x -> T.(x == diag_pos), npartials)
        partials = ForwardDiff.Partials{npartials, T}(D)
        v = ForwardDiff.Dual{tag, T, npartials}(v, partials)
    end
    return v
end

"""
    update_values!(x, dx)

Replace values of `x` in-place by `y`, leaving `x` with the values of y and the partials of `x`.
"""
@inline function update_values!(v::AbstractArray{<:Real}, next::AbstractArray{<:Real})
    # The ForwardDiff type is immutable, so to preserve the derivatives we do this little trick:
    @. v = v - value(v) + value(next)
end

"""
    update_values!(x, dx)

Replace values (for non-Real types, direct assignment)
"""
@inline function update_values!(v::AbstractArray{<:Any}, next::AbstractArray{<:Any})
    @. v = next
end

"""
Take value of AD.
"""
@inline function value(x)
    return ForwardDiff.value(x)
end

@inline function value(x::AbstractArray)
    return value.(x)
end

"""
    value(d::Dict)
Call value on all elements of some Dict.
"""
function value(d::AbstractDict)
    v = copy(d)
    for key in keys(v)
        v[key] = value(v[key])
    end
    return v
end

"""
Create a mapped array that produces only the values when indexed.

Only useful for AD arrays, otherwise it does nothing.
"""
@inline function as_value(x::AbstractArray)
    return mappedarray(value, x)
end

@inline function as_value(x::AbstractArray{X}) where X<:AbstractFloat
    return x
end

@inline function as_value(x)
    return x
end

"""
    get_entity_tag(basetag, entity)

Combine a base tag (which can be nothing) with a entity to get a tag that
captures base tag + entity tag for use with AD initialization.
"""
get_entity_tag(basetag, entity) = (basetag, entity)
get_entity_tag(::Nothing, entity) = entity

include("local_ad.jl")
include("sparsity.jl")
include("gradients.jl")
include("force_gradients.jl")
