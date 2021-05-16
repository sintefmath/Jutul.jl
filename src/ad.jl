# Primary variables

## Definition
function select_primary_variables(domain, system::TervSystem, formulation)
    return nothing
end

function number_of_units(model, ::TervPrimaryVariables)
    # By default, each primary variable exists on all cells of a discretized domain
    return number_of_cells(model.domain)
end

function number_of_degrees_of_freedom(model, pvars::TervPrimaryVariables)
    return number_of_units(model, pvars)*degrees_of_freedom_per_unit(pvars)
end

function degrees_of_freedom_per_unit(::ScalarPrimaryVariable)
    return 1
end

## Update functions

function absolute_increment_limit(::TervPrimaryVariables) nothing end
function relative_increment_limit(::TervPrimaryVariables) nothing end
function maximum_value(::TervPrimaryVariables) nothing end
function minimum_value(::TervPrimaryVariables) nothing end

function update_state!(state, p::TervPrimaryVariables, model, dx)
    names = get_names(p)
    nu = number_of_units(model, p)
    abs_max = absolute_increment_limit(p)
    rel_max = relative_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)

    for (index, nm) in enumerate(names)
        offset = nu*(index-1)
        v = state[Symbol(nm)] # TODO: Figure out this.
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
function limit_abs(dv, abs_change)
    dv = sign(dv)*min(abs(dv), abs_change)
end

function limit_abs(dv, ::Nothing) dv end

# Limit relative 
function limit_rel(v, dv, rel_change)
    dv = limit_abs(dv, rel_change*abs(v))
end

function limit_rel(v, dv, ::Nothing) dv end
# Lower bounds
function limit_upper(v, dv, maxval)
    if dv > 0 && v + dv > maxval
        dv = maxval - v
    end
    return dv
end

function limit_upper(v, dv, maxval::Nothing) dv end

# Upper bounds
function limit_lower(v, dv, minval)
    if dv < 0 && v + dv < minval
        dv = minval - v
    end
    return dv
end

function limit_lower(v, dv, minval::Nothing) dv end

function update_value(v, dv, arg...)
    return v + choose_increment(value(v), dv, arg...)
end


function get_names(v::TervPrimaryVariables)
    return [get_name(v)]
end

function get_symbol(v::TervPrimaryVariables)
    return v.symbol
end

function get_name(v::TervPrimaryVariables)
    return String(get_symbol(v))
end

function number_of_primary_variables(model)
    # TODO: Bit of a mess (number of primary variables, vs number of actual primary variables realized on grid. Fix.)
    return length(get_primary_variable_names(model))
end

## Initialization
function initialize_primary_variable_ad(state, model, pvar::ScalarPrimaryVariable, offset, npartials)
    name = get_name(pvar)
    state[name] = allocate_array_ad(state[name], diag_pos = offset + 1, context = model.context, npartials = npartials)
    return state
end

function initialize_primary_variable_value(state, model, pvar::ScalarPrimaryVariable, val::Union{Dict, AbstractFloat})
    n = number_of_degrees_of_freedom(model, pvar)
    name = get_name(pvar)
    if isa(val, Dict)
        val = val[name]
    end

    if isa(val, AbstractVector)
        V = deepcopy(val)
        @assert length(val) == n "Variable was neither scalar nor the expected dimension"
    else
        V = repeat([val], n)
    end
    state[name] = transfer(model.context, V)
    return state
end

"""
Convert a state containing variables as arrays of doubles
to a state where those arrays contain the same value as Dual types.
The dual type is currently taken from ForwardDiff.
"""
function convert_state_ad(model, state)
    context = model.context
    stateAD = deepcopy(state)
    vars = String.(keys(state))

    primary = get_primary_variables(model)
    # Loop over primary variables and set them to AD, with ones at the correct diagonal
    counts = map((x) -> degrees_of_freedom_per_unit(x), primary)
    n_partials = sum(counts)
    @debug "Found $n_partials primary variables."
    offset = 0
    for (i, pvar) in enumerate(primary)
        stateAD = initialize_primary_variable_ad(stateAD, model, pvar, offset, n_partials)
        offset += counts[i]
    end
    primary_names = get_primary_variable_names(model)
    secondary = setdiff(vars, primary_names)
    # Loop over secondary variables and initialize as AD with zero partials
    for s in secondary
        stateAD[s] = allocate_array_ad(stateAD[s], context = context, npartials = n_partials)
    end
    return stateAD
end

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

