# Primary variables

"""
Number of entities (e.g. Cells, Faces) a variable is defined on.
By default, each primary variable exists on all cells of a discretized domain

"""
number_of_entities(model, pv::TervVariables) = count_entities(model.domain, associated_entity(pv))

"""
The entity a variable is associated with, and can hold partial derivatives with respect to.
"""
associated_entity(::TervVariables) = Cells()

"""
Total number of degrees of freedom for a model, over all primary variables and all entities.
"""
function number_of_degrees_of_freedom(model::TervModel)
    ndof = 0
    for (pkey, pvar) in get_primary_variables(model)
        ndof += number_of_degrees_of_freedom(model, pvar)
    end
    return ndof
end

function number_of_degrees_of_freedom(model::TervModel, u::TervUnit)
    ndof = degrees_of_freedom_per_entity(model, u)*count_active_entities(model.domain, u)
    return ndof
end

function degrees_of_freedom_per_entity(model::TervModel, u::TervUnit)
    ndof = 0
    for pvar in values(get_primary_variables(model))
        if associated_entity(pvar) == u
            ndof += degrees_of_freedom_per_entity(model, pvar)
        end
    end
    return ndof
end

function number_of_degrees_of_freedom(model, pvars::TervVariables)
    e = associated_entity(pvars)
    n = count_active_entities(model.domain, e)
    m = degrees_of_freedom_per_entity(model, pvars)
    return n*m
end

function value_dim(model, pvars::TervVariables)
    return (values_per_entity(model, pvars), number_of_entities(model, pvars))
end

"""
Number of independent primary variables / degrees of freedom per computational entity.
"""
degrees_of_freedom_per_entity(model, ::ScalarVariable) = 1
"""
Constant variables hold no degrees of freedom.
"""
degrees_of_freedom_per_entity(model, ::ConstantVariables) = 0

"""
Number of values held by a primary variable. Normally this is equal to the number of degrees of freedom,
but some special primary variables are most conveniently defined by having N values and N-1 independent variables.
"""
values_per_entity(model, u::TervVariables) = degrees_of_freedom_per_entity(model, u)
## Update functions
"""
Absolute allowable change for variable during a nonlinear update.
"""
absolute_increment_limit(::TervVariables) = nothing

"""
Relative allowable change for variable during a nonlinear update.
A variable with value |x| and relative limit 0.2 cannot change more
than |x|*0.2.
"""
relative_increment_limit(::TervVariables) = nothing

"""
Upper (inclusive) limit for variable.
"""
maximum_value(::TervVariables) = nothing

"""
Lower (inclusive) limit for variable.
"""
minimum_value(::TervVariables) = nothing

function update_primary_variable!(state, p::TervVariables, state_symbol, model, dx)
    names = get_names(p)
    entity = associated_entity(p)
    active = active_entities(model.domain, entity)

    nu = length(active)
    abs_max = absolute_increment_limit(p)
    rel_max = relative_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)
    scale = variable_scale(p)

    for (index, nm) in enumerate(names)
        offset = nu*(index-1)
        v = state[state_symbol]
        @tullio v[active[i]] = update_value(v[active[i]], dx[i+offset], abs_max, rel_max, minval, maxval, scale)
    end
end

@inline function choose_increment(v::F, dv::F, abs_change = nothing, rel_change = nothing, minval = nothing, maxval = nothing, scale = nothing) where {F<:AbstractFloat}
    dv = scale_increment(dv, scale)
    dv = limit_abs(dv, abs_change)
    dv = limit_rel(v, dv, rel_change)
    dv = limit_lower(v, dv, minval)
    dv = limit_upper(v, dv, maxval)
    return dv
end
# Limit absolute
limit_abs(dv, abs_change) = sign(dv)*min(abs(dv), abs_change)
limit_abs(dv, ::Nothing) = dv
# Limit relative 
limit_rel(v, dv, rel_change) = limit_abs(dv, rel_change*abs(v))
limit_rel(v, dv, ::Nothing) = dv
# Lower bounds
function limit_upper(v, dv, maxval)
    if dv > 0 && v + dv > maxval
        dv = maxval - v
    end
    return dv
end
limit_upper(v, dv, maxval::Nothing) = dv

# Upper bounds
@inline function limit_lower(v, dv, minval)
    if dv < 0 && v + dv < minval
        dv = minval - v
    end
    return dv
end
limit_lower(v, dv, minval::Nothing) = dv

# Scaling
scale_increment(dv, scale) = dv*scale
scale_increment(dv, ::Nothing) = dv

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

variable_scale(::TervVariables) = nothing

## Initialization
function initialize_primary_variable_ad!(state, model, pvar, arg...; offset = 0, kwarg...)
    diag_value = variable_scale(pvar)
    if isnothing(diag_value)
        diag_value = 1.0
    end
    initialize_variable_ad(state, model, pvar, arg..., offset + 1; diag_value = diag_value, kwarg...)
end

function initialize_secondary_variable_ad!(state, model, pvar, arg...; kwarg...)
    initialize_variable_ad(state, model, pvar, arg..., NaN; kwarg...)
end

function initialize_variable_ad(state, model, pvar, symb, npartials, diag_pos; kwarg...)
    state[symb] = allocate_array_ad(state[symb], diag_pos = diag_pos, context = model.context, npartials = npartials; kwarg...)
    return state
end

function initialize_variable_value(model, pvar, val; perform_copy = true)
    nu = number_of_entities(model, pvar)
    nv = values_per_entity(model, pvar)
    
    if isa(pvar, ScalarVariable)
        @assert length(val) == nu "Expected $nu entries, but got $(length(val)) for $(typeof(pvar))"
        # Type-assert that this should be scalar, with a vector input
        val::AbstractVector
    else
        nm = length(val)
        @assert length(val) == nm "Passed value had $nm entries, expected $(nu*nv)"
        n, m, = size(val)
        @assert n == nv "Passed value had $n rows, expected $nv"
        @assert m == nu "Passed value had $m rows, expected $nu"
    end
    if perform_copy
        val = deepcopy(val)
    end
    return transfer(model.context, val)
end

default_value(model, variable) = 0.0

function initialize_variable_value!(state, model, pvar, symb, val; kwarg...)
    state[symb] = initialize_variable_value(model, pvar, val; kwarg...)
    return state
end

function initialize_variable_value!(state, model, pvar, symb, val::AbstractDict; need_value = true)
    if haskey(val, symb)
        value = val[symb]
    elseif need_value
        k = keys(val)
        error("The key $symb must be present to initialize the state. Provided symbols in initialization Dict: $k")
    else
        # We do not really need to initialize this, as it will be updated elsewhere.
        value = default_value(model, pvar)
    end
    return initialize_variable_value!(state, model, pvar, symb, value)
end

# Scalar primary variables
function initialize_variable_value!(state, model, pvar::ScalarVariable, symb::Symbol, val::Number)
    V = repeat([val], number_of_entities(model, pvar))
    return initialize_variable_value!(state, model, pvar, symb, V)
end

"""
Initializer for the value of non-scalar primary variables
"""
function initialize_variable_value!(state, model, pvar::GroupedVariables, symb::Symbol, val::AbstractVector)
    n = values_per_entity(model, pvar)
    t = typeof(pvar)
    @assert length(val) == n "Variable $t should have initializer of length $n"
    V = repeat(val, 1, number_of_entities(model, pvar))
    return initialize_variable_value!(state, model, pvar, symb, V)
end

function initialize_variable_value!(state, model, pvar::GroupedVariables, symb::Symbol, val::Number)
    n = values_per_entity(model, pvar)
    return initialize_variable_value!(state, model, pvar, symb, repeat([val], n))
end

# Specific variable implementations that are generic for many types of system follow
degrees_of_freedom_per_entity(model, v::FractionVariables) =  values_per_entity(model, v) - 1
maximum_value(::FractionVariables) = 1.0
minimum_value(::FractionVariables) = 0.0


function initialize_primary_variable_ad!(state, model, pvar::FractionVariables, state_symbol, npartials; kwarg...)
    n = values_per_entity(model, pvar)
    v = state[state_symbol]
    state[state_symbol] = unit_sum_init(v, model, npartials, n; kwarg...)
    return state
end

function unit_sum_init(v, model, npartials, N; offset = 0, kwarg...)
    # nph - 1 primary variables, with the last saturation being initially zero AD
    dp = vcat((1:N-1) .+ offset, 0)
    v = allocate_array_ad(v, diag_pos = dp, context = model.context, npartials = npartials; kwarg...)
    for i in 1:size(v, 2)
        v[end, i] = 1 - sum(v[1:end-1, i])
    end
    return v
end

function update_primary_variable!(state, p::FractionVariables, state_symbol, model, dx)
    s = state[state_symbol]
    unit_sum_update!(s, p, model, dx)
end

function unit_sum_update!(s, p, model, dx)
    nf, nu = value_dim(model, p)
    abs_max = absolute_increment_limit(p)
    maxval, minval = maximum_value(p), minimum_value(p)
    active_cells = active_entities(model.domain, Cells())
    if nf == 2
        maxval = min(1 - minval, maxval)
        minval = max(minval, maxval - 1)
        @inbounds for (i, cell) in enumerate(active_cells)
            v = value(s[1, cell])
            dv = dx[i]
            dv = choose_increment(v, dv, abs_max, nothing, minval, maxval)
            s[1, cell] += dv
            s[2, cell] -= dv
        end
    else
        if false
            # Preserve direction
            for cell in active_cells
                w = 1.0
                # First pass: Find the relaxation factors that keep all fractions in [0, 1]
                # and obeying the maximum change targets
                dlast0 = 0
                @inbounds for i = 1:(nf-1)
                    v = value(s[i, cell])
                    dv0 = dx[cell + (i-1)*nu]
                    dv = choose_increment(v, dv0, abs_max, nothing, minval, maxval)
                    dlast0 -= dv0
                    w = pick_relaxation(w, dv, dv0)
                end
                # Do the same thing for the implicit update of the last value
                dlast = choose_increment(value(s[nf, cell]), dlast0, abs_max, nothing, minval, maxval)
                w = pick_relaxation(w, dlast, dlast0)

                @inbounds for i = 1:(nf-1)
                    s[i, cell] += w*dx[cell + (i-1)*nu]
                end
                @inbounds s[nf, cell] += w*dlast0
            end
        else
            # Preserve update magnitude
            for cell in active_cells
                unit_update_local(s, dx, nf, nu, abs_max, minval, maxval)
            end
        end
    end
end

function unit_update_local(s, dx, nf, nu, abs_max, minval, maxval)
    # First pass: Find the relaxation factors that keep all fractions in [0, 1]
    # and obeying the maximum change targets
    dlast0 = 0
    @inbounds for i = 1:(nf-1)
        v = value(s[i, cell])
        dv = dx[cell + (i-1)*nu]
        dv = choose_increment(v, dv, abs_max, nothing, minval, maxval)
        s[i, cell] += dv
        dlast0 -= dv
    end
    # Do the same thing for the implicit update of the last value
    dlast = choose_increment(value(s[nf, cell]), dlast0, abs_max, nothing, minval, maxval)
    s[nf, cell] += dlast
    if dlast != dlast0
        t = 0.0
        for i = 1:nf
            @inbounds t += s[i, cell]
        end
        for i = 1:nf
            @inbounds s[i, cell] /= t
        end
    end
end

function pick_relaxation(w, dv, dv0)
    # dv0*w = dv -> w = dv/dv0
    r = dv/dv0
    if dv0 != 0
        w = min(w, r)
    end
    return w
end

function values_per_entity(model, var::ConstantVariables)
    c = var.constants
    if var.single_entity
        n = length(c)
    else
        n = size(c, 1)
    end
end
associated_entity(model, var::ConstantVariables) = var.entity

function transfer(context, v::ConstantVariables)
    constants = transfer(context, v.constants)
    return ConstantVariables(constants, v.entity, single_entity = v.single_entity)
end

function initialize_variable_value(model, var::ConstantVariables, val; perform_copy = true)
    # Ignore initializer since we already know the constants
    nu = number_of_entities(model, var)
    if var.single_entity
        # it = index_type(model.context)
        # use instance as view to avoid allocating lots of copies
        var_val = ConstantWrapper(var.constants, nu)# view(var.constants, :, ones(it, nu))
    else
        # We have all the values we need ready.
        var_val = var.constants
    end
    return var_val
end

function initialize_secondary_variable_ad!(state, model, var::ConstantVariables, arg...; kwarg...)
    # Do nothing. There is no need to add AD.
    return state
end

function initialize_primary_variable_ad!(state, model, var::ConstantVariables, sym, arg...; kwarg...)
    error("$sym is declared to be constants - cannot be primary variables.")
end

default_value(model, variable::ConstantVariables) = nothing

