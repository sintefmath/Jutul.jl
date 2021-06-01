export CompactAutoDiffCache, as_value

abstract type TervAutoDiffCache end
struct CompactAutoDiffCache{I, ∂x} <: TervAutoDiffCache where {I <: Integer, ∂x <: Real}
    entries
    unit
    jacobian_positions
    equations_per_unit::I
    number_of_units::I
    npartials::I
    function CompactAutoDiffCache(equations_per_unit, n_units, npartials_or_model = 1; unit = Cells(), context = DefaultContext(), tag = nothing, kwarg...)
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
        pos = zeros(I, equations_per_unit*npartials, n_units)
        pos = transfer(context, pos)
        new{I, D}(entries, unit, pos, equations_per_unit, n_units, npartials)
    end
end

@inline function number_of_units(c::TervAutoDiffCache) c.number_of_units end

@inline function get_entries(e::TervEquation)
    return get_entries(e.equation)
end

@inline function get_entries(c::CompactAutoDiffCache)
    return c.entries
end

@inline function get_entry(c::CompactAutoDiffCache{I, D}, index, eqNo, entries)::D where {I, D}
    @inbounds entries[eqNo, index]
end

@inline function get_value(c::CompactAutoDiffCache, arg...)
    value(get_entry(c, arg...))
end

@inline function get_partial(c::CompactAutoDiffCache, index, eqNo = 1, partial_index = 1)
    get_entry(c, index, eqNo).partials[partial_index]
end

@inline function get_partials(c::CompactAutoDiffCache, index, eqNo = 1)
    get_entry(c, index, eqNo).partials
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
                                                                        new_value = get_partial(c, index, eqNo, partial_index),
                                                                        pos = c.jacobian_positions)
    @inbounds nzval[get_jacobian_pos(c, index, eqNo, partial_index, pos)] = new_value
end

function update_linearized_system_subset!(nz, r, model, cache::TervAutoDiffCache)
    nu = cache.number_of_units
    entries = cache.entries
    pos = cache.jacobian_positions
    Threads.@threads for i in 1:nu
        for e in 1:cache.equations_per_unit
            # Note: The residual part needs to be fixed for non-standard alignments
            a = get_entry(cache, i, e, entries)
            @inbounds r[i + nu*(e-1)] = a.value
            for d = 1:cache.npartials
                @inbounds ∂ = a.partials[d]
                update_jacobian_entry!(nz, cache, i, e, d, ∂, pos)
            end
        end
    end
end

function update_linearized_system_subset!(nz, r::Nothing, model, cache::TervAutoDiffCache)
    nu = cache.number_of_units
    entries = cache.entries
    Threads.@threads for i in 1:nu
        for e in 1:cache.equations_per_unit
            # Note: The residual part needs to be fixed for non-standard alignments
            a = get_entry(cache, i, e, entries)
            for d = 1:cache.npartials
                update_jacobian_entry!(nz, cache, i, e, d, a.partials[d])
            end
        end
    end
end

function diagonal_alignment!(cache, arg...; eq_index = 1:cache.number_of_units, kwarg...)
    injective_alignment!(cache, arg...; target_index = eq_index, source_index = eq_index, kwarg...)
end

function injective_alignment!(cache::TervAutoDiffCache, jac, unit, layout;
                                    target_index = 1:cache.number_of_units,
                                    source_index = 1:cache.number_of_units,
                                    target_offset = 0,
                                    source_offset = 0)
    unit::TervUnit
    cache.unit::TervUnit
    if unit == cache.unit
        nu, ne, np = ad_dims(cache)
        nix = length(target_index)
        N = length(source_index)
        @assert length(target_index) == N
        for index in 1:N
            target = target_index[index]
            source = source_index[index]
            for e in 1:ne
                for d = 1:np
                    pos = find_jac_position(jac, target + target_offset, source + source_offset, e, d, nix, nu, ne, np, layout)
                    set_jacobian_pos!(cache, target, e, d, pos)
                end
            end
        end
    end
end


# Primary variables
function number_of_units(model, pv::TervVariables)
    # By default, each primary variable exists on all cells of a discretized domain
    return count_units(model.domain, associated_unit(pv))
end

function associated_unit(::TervVariables)
    # The default unit for all primary variables is Cells()
    return Cells()
end

function number_of_degrees_of_freedom(model::TervModel)
    ndof = 0
    for (pkey, pvar) in get_primary_variables(model)
        ndof += number_of_degrees_of_freedom(model, pvar)
    end
    return ndof
end

function number_of_degrees_of_freedom(model::TervModel, u::TervUnit)
    ndof = degrees_of_freedom_per_unit(model, u)*count_units(model.domain, u)
    return ndof
end

function degrees_of_freedom_per_unit(model::TervModel, u::TervUnit)
    ndof = 0
    for pvar in values(get_primary_variables(model))
        if associated_unit(pvar) == u
            ndof += degrees_of_freedom_per_unit(model, pvar)
        end
    end
    return ndof
end

function number_of_degrees_of_freedom(model, pvars::TervVariables)
    return number_of_units(model, pvars)*degrees_of_freedom_per_unit(model, pvars)
end

function value_dim(model, pvars::TervVariables)
    return (values_per_unit(model, pvars), number_of_units(model, pvars))
end

"""
Number of independent primary variables / degrees of freedom per computational unit.
"""
function degrees_of_freedom_per_unit(model, ::ScalarVariable)
    return 1
end
"""
Number of values held by a primary variable. Normally this is equal to the number of degrees of freedom,
but some special primary variables are most conveniently defined by having N values and N-1 independent variables.
"""
function values_per_unit(model, u::TervVariables)
    return degrees_of_freedom_per_unit(model, u)
end

## Update functions

function absolute_increment_limit(::TervVariables) nothing end
function relative_increment_limit(::TervVariables) nothing end
function maximum_value(::TervVariables) nothing end
function minimum_value(::TervVariables) nothing end

function update_primary_variable!(state, p::TervVariables, state_symbol, model, dx)
    names = get_names(p)
    nu = number_of_units(model, p)
    abs_max = absolute_increment_limit(p)
    rel_max = relative_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)

    for (index, nm) in enumerate(names)
        offset = nu*(index-1)
        v = state[state_symbol]
        dv = view(dx, (1:nu) .+ offset)
        @. v = update_value(v, dv, abs_max, rel_max, minval, maxval)
    end
end

@inline function choose_increment(v::F, dv::F, abs_change, rel_change, minval, maxval) where {F<:AbstractFloat}
    dv = limit_abs(dv, abs_change)
    dv = limit_rel(v, dv, rel_change)
    dv = limit_lower(v, dv, minval)
    dv = limit_upper(v, dv, maxval)
    return dv
end


# Limit absolute
@inline function limit_abs(dv, abs_change)
    dv = sign(dv)*min(abs(dv), abs_change)
end

@inline function limit_abs(dv, ::Nothing) dv end

# Limit relative 
@inline function limit_rel(v, dv, rel_change)
    dv = limit_abs(dv, rel_change*abs(v))
end

@inline function limit_rel(v, dv, ::Nothing) dv end
# Lower bounds
function limit_upper(v, dv, maxval)
    if dv > 0 && v + dv > maxval
        dv = maxval - v
    end
    return dv
end

@inline function limit_upper(v, dv, maxval::Nothing) dv end

# Upper bounds
@inline function limit_lower(v, dv, minval)
    if dv < 0 && v + dv < minval
        dv = minval - v
    end
    return dv
end

@inline function limit_lower(v, dv, minval::Nothing) dv end

@inline function update_value(v, dv, arg...)
    return v + choose_increment(value(v), dv, arg...)
end


function get_names(v::TervVariables)
    return [get_name(v)]
end

function get_symbol(v::TervVariables)
    return Symbol(typeof(v))
end

function get_name(v::TervVariables)
    return String(get_symbol(v))
end

function number_of_primary_variables(model)
    # TODO: Bit of a mess (number of primary variables, vs number of actual primary variables realized on grid. Fix.)
    return length(get_primary_variable_names(model))
end

## Initialization
function initialize_primary_variable_ad!(arg...; offset = 0, kwarg...)
    initialize_variable_ad(arg..., offset + 1; kwarg...)
end

function initialize_secondary_variable_ad(arg...; kwarg...)
    initialize_variable_ad(arg..., NaN; kwarg...)
end

function initialize_variable_ad(state, model, pvar, symb, npartials, diag_pos; kwarg...)
    state[symb] = allocate_array_ad(state[symb], diag_pos = diag_pos, context = model.context, npartials = npartials; kwarg...)
    return state
end

function initialize_variable_value(model, pvar, val; perform_copy = true)
    nu = number_of_units(model, pvar)
    nv = values_per_unit(model, pvar)
    
    if isa(pvar, ScalarVariable)
        @assert length(val) == nu
        # Type-assert that this should be scalar, with a vector input
        val::AbstractVector
    else
        @assert size(val, 1) == nv
        @assert size(val, 2) == nu
    end
    if perform_copy
        val = deepcopy(val)
    end
    return transfer(model.context, val)
end

function initialize_variable_value!(state, model, pvar, symb, val; kwarg...)
    state[symb] = initialize_variable_value(model, pvar, val; kwarg...)
    return state
end

function initialize_variable_value!(state, model, pvar, symb, val::AbstractDict; need_value = true)
    if haskey(val, symb)
        value = val[symb]
    elseif need_value
        k = keys(val)
        error("The key $symb must be present to initialize the state. Found symbols: $k")
    else
        # We do not really need to initialize this, as it will be updated elsewhere.
        value = 0.0
    end
    return initialize_variable_value!(state, model, pvar, symb, value)
end

# Scalar primary variables
function initialize_variable_value!(state, model, pvar::ScalarVariable, symb::Symbol, val::Number)
    V = repeat([val], number_of_units(model, pvar))
    return initialize_variable_value!(state, model, pvar, symb, V)
end

# Non-scalar primary variables
function initialize_variable_value!(state, model, pvar::GroupedVariables, symb::Symbol, val::AbstractVector)
    n = values_per_unit(model, pvar)
    t = typeof(pvar)
    @assert length(val) == n "Variable $t should have initializer of length $n"
    V = repeat(val, 1, number_of_units(model, pvar))
    return initialize_variable_value!(state, model, pvar, symb, V)
end

function initialize_variable_value!(state, model, pvar::GroupedVariables, symb::Symbol, val::Number)
    n = values_per_unit(model, pvar)
    return initialize_variable_value!(state, model, pvar, symb, repeat([val], n))
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
    if isnothing(tag)
        @debug "Setting up primary variables..."
    else
        @debug "$tag: Setting up primary variables..."
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
            @debug "Variable group:\n\t$(n_units) $(typeof(u)) with $n_partials partial derivatives each ($(n_partials*n_units) total)."
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
        @debug "→ $pkey:\n\t$n_local of $n_partials partials on all $(typeof(u)), covers $(offset+1) → $(offset + n_local)"
        stateAD = initialize_primary_variable_ad!(stateAD, model, pvar, pkey, n_partials, tag = t, offset = offset, context = context)
        offset += n_local
        total_number_of_partials += n_local
    end
    @debug "Primary variables set up: $(number_of_degrees_of_freedom(model)) degrees of freedom\n\t → $total_number_of_partials distinct primary variables over $total_number_of_groups different units."
    secondary = get_secondary_variables(model)
    # Loop over secondary variables and initialize as AD with zero partials
    @debug "Setting up secondary variables..."
    for (skey, svar) in secondary
        u = associated_unit(svar)
        @debug "$skey: Defined on $(typeof(u))"

        t = get_unit_tag(tag, u)
        n_partials = degrees_of_freedom_per_unit(model, u)
        stateAD = initialize_secondary_variable_ad(stateAD, model, svar, skey, n_partials, tag = t, context = context)
    end
    return stateAD
end

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

function allocate_array_ad(v::AbstractVector; kwarg...)
    # create a copy of a vector as AD
    v_AD = allocate_array_ad(length(v); kwarg...)
    update_values!(v_AD, v)
end

# Allocators 
function allocate_array_ad(v::AbstractMatrix; kwarg...)
    # create a copy of a vector as AD
    v_AD = allocate_array_ad(size(v)...; kwarg...)
    update_values!(v_AD, v)
end

function get_ad_unit_scalar(v::T, npartials, diag_pos = nothing; tag = nothing) where {T<:Real}
    # Get a scalar, with a given number of zero derivatives. A single entry can be specified to be non-zero
    if npartials > 0
        v = ForwardDiff.Dual{tag}(v, ntuple(x -> T.(x == diag_pos), npartials))
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

@inline function as_value(x)
    mappedarray(value, x)
end

function get_unit_tag(basetag, unit)
    utag = Symbol(typeof(unit))
    if !isnothing(basetag)
        utag = Symbol(String(utag)*"∈"*String(basetag))
    end
    utag
end
