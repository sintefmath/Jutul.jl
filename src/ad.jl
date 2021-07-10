export CompactAutoDiffCache, as_value, TervAutoDiffCache

"""
An AutoDiffCache is a type that holds both a set of AD values and a map into some
global Jacobian.
"""
abstract type TervAutoDiffCache end
"""
Cache that holds an AD vector/matrix together with their positions.
"""
struct CompactAutoDiffCache{I, ∂x} <: TervAutoDiffCache where {I <: Integer, ∂x <: Real}
    entries
    unit
    jacobian_positions
    equations_per_unit::I
    number_of_units::I
    npartials::I
    function CompactAutoDiffCache(equations_per_unit, n_units, npartials_or_model = 1; 
                                                        unit = Cells(),
                                                        context = DefaultContext(),
                                                        tag = nothing,
                                                        n_units_pos = nothing,
                                                        kwarg...)
        if isa(npartials_or_model, TervModel)
            model = npartials_or_model
            npartials = degrees_of_freedom_per_unit(model, unit)
        else
            npartials = npartials_or_model
        end
        npartials::Integer

        I = index_type(context)
        # Storage for AD variables
        t = get_unit_tag(tag, unit)
        entries = allocate_array_ad(equations_per_unit, n_units, context = context, npartials = npartials, tag = t; kwarg...)
        D = eltype(entries)
        # Position in sparse matrix - only allocated, then filled in later.
        # Since partials are all fetched together with the value, we make partials the fastest index.
        if isnothing(n_units_pos)
            # This can be overriden - if a custom assembly is planned.
            n_units_pos = n_units
        end
        pos = zeros(I, equations_per_unit*npartials, n_units_pos)
        pos = transfer(context, pos)
        new{I, D}(entries, unit, pos, equations_per_unit, n_units, npartials)
    end
end

"""
Get number of units a cache is defined on.
"""
@inline function number_of_units(c::TervAutoDiffCache) c.number_of_units end

"""
Get the entries of the main autodiff cache for an equation.

Note: This only gets the .equation field's entries.
"""
@inline function get_entries(e::TervEquation)
    return get_entries(e.equation)
end

"""
Get entries of autodiff cache. Entries are AD vectors that hold values and derivatives.
"""
@inline function get_entries(c::CompactAutoDiffCache)
    return c.entries
end

@inline function get_entry(c::CompactAutoDiffCache{I, D}, index, eqNo, entries)::D where {I, D}
    @inbounds entries[eqNo, index]
end

@inline function get_value(c::CompactAutoDiffCache, arg...)
    value(get_entry(c, arg...))
end

@inline function get_jacobian_pos(c::CompactAutoDiffCache{I}, index, eqNo, partial_index, pos)::I where {I}
    @inbounds pos[(eqNo-1)*c.npartials + partial_index, index]
end

@inline function set_jacobian_pos!(c::CompactAutoDiffCache, index, eqNo, partial_index, pos)
    c.jacobian_positions[(eqNo-1)*c.npartials + partial_index, index] = pos
end

@inline function ad_dims(cache::CompactAutoDiffCache{I, D})::Tuple{I, I, I} where {I, D}
    return (cache.number_of_units, cache.equations_per_unit, cache.npartials)
end

@inline function update_jacobian_entry!(nzval, c::CompactAutoDiffCache, index, eqNo, partial_index, 
                                                                        new_value,
                                                                        pos = c.jacobian_positions)
    @inbounds nzval[get_jacobian_pos(c, index, eqNo, partial_index, pos)] = new_value
end

function fill_equation_entries!(nz, r, model, cache::TervAutoDiffCache)
    nu, ne, np = ad_dims(cache)
    entries = cache.entries
    jp = cache.jacobian_positions
    @threads for i in 1:nu
        for e in 1:ne
            a = get_entry(cache, i, e, entries)
            @inbounds r[i + nu*(e-1)] = a.value
            for d = 1:cache.npartials
                apos = get_jacobian_pos(cache, i, e, d, jp)
                @inbounds nz[apos] = a.partials[d]
            end
        end
    end
end

function fill_equation_entries!(nz, r::Nothing, model, cache::TervAutoDiffCache)
    nu, ne, np = ad_dims(cache)
    entries = cache.entries
    jp = cache.jacobian_positions
    @threads for i in 1:nu
        for e in 1:ne
            a = get_entry(cache, i, e, entries)
            for d = 1:np
                @inbounds ∂ = a.partials[d]
                # TODO:
                # This part is type unstable, for some reason.
                # update_jacobian_entry!(nz, cache, i, e, d, ∂)
                # Use more verbose/inner version instead:
                apos = get_jacobian_pos(cache, i, e, d, jp)
                @inbounds nz[apos] = a.partials[d]
            end
        end
    end
end

function diagonal_alignment!(cache, arg...; eq_index = 1:cache.number_of_units, kwarg...)
    injective_alignment!(cache, arg...; target_index = eq_index, source_index = eq_index, kwarg...)
end

function injective_alignment!(cache::TervAutoDiffCache, jac, unit, context;
                                    target_index = 1:cache.number_of_units,
                                    source_index = 1:cache.number_of_units,
                                    number_of_units_source = nothing,
                                    number_of_units_target = nothing,
                                    target_offset = 0,
                                    source_offset = 0)
    unit::TervUnit
    cache.unit::TervUnit
    layout = matrix_layout(context)
    if unit == cache.unit
        nu_c, ne, np = ad_dims(cache)
        if isnothing(number_of_units_source)
            nu_s = nu_c
        else
            nu_s = number_of_units_source
        end
        if isnothing(number_of_units_target)
            nu_t = length(target_index)
        else
            nu_t = number_of_units_target
        end
        N = length(source_index)
        @assert length(target_index) == N
        do_injective_alignment!(cache, jac, target_index, source_index, nu_t, nu_s, ne, np, target_offset, source_offset, layout)
    end
end

function do_injective_alignment!(cache, jac, target_index, source_index, nu_t, nu_s, ne, np, target_offset, source_offset, layout)
    for index in 1:length(source_index)
        target = target_index[index]
        source = source_index[index]
        for e in 1:ne
            for d = 1:np
                pos = find_jac_position(jac, target + target_offset, source + source_offset, e, d, 
                nu_t, nu_s,
                ne, np,
                layout)
                set_jacobian_pos!(cache, index, e, d, pos)
            end
        end
    end
end

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
    last_unit = nothing
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
        u = associated_unit(pvar)
        # Number of partials for this unit
        n_partials = degrees_of_freedom_per_unit(model, u)
        if last_unit != u
            n_units = count_units(model.domain, u)
            outstr *= "Variable group:\n\t$(n_units) $(typeof(u)) with $n_partials partial derivatives each ($(n_partials*n_units) total).\n"
            # Note: We assume that the variables are sorted by units.
            # This is asserted for in the model constructor.
            last_unit = u
            # Reset the offset to zero, since we have entered a new group.
            offset = 0
            total_number_of_groups += 1
        end
        # Number of partials this primary variable contributes
        n_local = degrees_of_freedom_per_unit(model, pvar)
        t = get_unit_tag(tag, u)
        outstr *= "→ $pkey:\n\t$n_local of $n_partials partials on all $(typeof(u)), covers $(offset+1) → $(offset + n_local)\n"
        stateAD = initialize_primary_variable_ad!(stateAD, model, pvar, pkey, n_partials, tag = t, offset = offset, context = context)
        offset += n_local
        total_number_of_partials += n_local
    end
    outstr *= "Primary variables set up: $(number_of_degrees_of_freedom(model)) degrees of freedom\n\t → $total_number_of_partials distinct primary variables over $total_number_of_groups different units.\n"
    secondary = get_secondary_variables(model)
    # Loop over secondary variables and initialize as AD with zero partials
    outstr *= "Setting up secondary variables...\n"
    for (skey, svar) in secondary
        u = associated_unit(svar)
        outstr *= "\t$skey: Defined on $(typeof(u))\n"

        t = get_unit_tag(tag, u)
        n_partials = degrees_of_freedom_per_unit(model, u)
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
- `diag_pos = nothing`: Indices of where to put units on the diagonal (if any)

Other keyword arguments are passed onto `get_ad_unit_scalar`.

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
function allocate_array_ad(n::R...; context::TervContext = DefaultContext(), diag_pos = nothing, npartials = 1, kwarg...) where {R<:Integer}
    # allocate a n length zero vector with space for derivatives
    T = float_type(context)
    if npartials == 0
        A = allocate_array(context, T(0), n...)
    else
        if isa(diag_pos, AbstractVector)
            @assert n[1] == length(diag_pos) "diag_pos must be specified for all columns."
            d = map(x -> get_ad_unit_scalar(T(0.0), npartials, x; kwarg...), diag_pos)
            A = allocate_array(context, d, 1, n[2:end]...)
        else
            d = get_ad_unit_scalar(T(0.0), npartials, diag_pos; kwarg...)
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
    get_ad_unit_scalar(v::Real, npartials, diag_pos = nothing; <keyword_arguments>)

Get scalar with partial derivatives as AD instance.

# Arguments
- `v::Real`: Value of AD variable.
- `npartials`: Number of partial derivatives each AD instance holds.
- `diag_pos` = nothing: Position(s) of where to set 1 as the partial derivative instead of zero.

# Keyword arguments
- `tag = nothing`: Tag for AD instance. Two AD values of the different tag cannot interoperate to avoid perturbation confusion (see ForwardDiff documentation).
"""
function get_ad_unit_scalar(v::T, npartials, diag_pos = nothing; tag = nothing) where {T<:Real}
    # Get a scalar, with a given number of zero derivatives. A single entry can be specified to be non-zero
    if npartials > 0
        v = ForwardDiff.Dual{tag}(v, ntuple(x -> T.(x == diag_pos), npartials))
    end
    return v
end

"""
    update_values!(x, dx)

Replace values of `x` in-place by `y`, leaving `x` with the avlues of y and the partials of `x`.
"""
@inline function update_values!(v::AbstractArray, next::AbstractArray)
    # The ForwardDiff type is immutable, so to preserve the derivatives we do this little trick:
    @. v = v - value(v) + value(next)
end

"""
Take value of AD.
"""
@inline function value(x)
    return ForwardDiff.value(x)
end

"""
    value(d::Dict)
Call value on all elements of some Dict.
"""
function value(d::AbstractDict)
    v = copy(d)
    for key in keys(v)
        v[key] = value.(v[key])
    end
    return v
end

"""
Create a mapped array that produces only the values when indexed.

Only useful for AD arrays, otherwise it does nothing.
"""
@inline function as_value(x)
    mappedarray(value, x)
end

"""
Combine a base tag (which can be nothing) with a unit to get a tag that
captures base tag + unit tag for use with AD initialization.
"""
function get_unit_tag(basetag, unit)
    utag = Symbol(typeof(unit))
    if !isnothing(basetag)
        utag = Symbol(String(utag)*"∈"*String(basetag))
    end
    utag
end

