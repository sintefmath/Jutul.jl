const MINIMUM_SAT_RELAX = 1e-6
"""
Number of entities (e.g. Cells, Faces) a variable is defined on.
By default, each primary variable exists on all cells of a discretized domain

"""
number_of_entities(model, pv::JutulVariables) = count_entities(model.domain, associated_entity(pv))

"""
The entity a variable is associated with, and can hold partial derivatives with respect to.
"""
associated_entity(::JutulVariables) = Cells()

"""
Total number of degrees of freedom for a model, over all primary variables and all entities.
"""
function number_of_degrees_of_freedom(model::JutulModel)
    ndof = 0
    for (pkey, pvar) in get_primary_variables(model)
        ndof += number_of_degrees_of_freedom(model, pvar)
    end
    return ndof
end

export number_of_values
"""
Total number of values for a model, for a given type of variables over all entities
"""
function number_of_values(model, type = :primary)
    ndof = 0
    for (pkey, pvar) in pairs(get_variables_by_type(model, type))
        ndof += number_of_values(model, pvar)
    end
    return ndof
end

function number_of_degrees_of_freedom(model::JutulModel, u::JutulEntity)
    ndof = degrees_of_freedom_per_entity(model, u)*count_active_entities(model.domain, u, for_variables = true)
    return ndof
end

function degrees_of_freedom_per_entity(model::JutulModel, u::JutulEntity)
    ndof = 0
    for pvar in values(get_primary_variables(model))
        if associated_entity(pvar) == u
            ndof += degrees_of_freedom_per_entity(model, pvar)
        end
    end
    return ndof
end

function number_of_degrees_of_freedom(model, pvars::JutulVariables)
    e = associated_entity(pvars)
    n = count_active_entities(model.domain, e, for_variables = true)
    m = degrees_of_freedom_per_entity(model, pvars)
    return n*m
end

number_of_values(model, pvars::JutulVariables) = prod(value_dim(model, pvars))

function value_dim(model, pvars::JutulVariables)
    return (values_per_entity(model, pvars), number_of_entities(model, pvars))
end

"""
Number of independent primary variables / degrees of freedom per computational entity.
"""
degrees_of_freedom_per_entity(model, ::ScalarVariable) = 1

"""
Number of values held by a primary variable. Normally this is equal to the number of degrees of freedom,
but some special primary variables are most conveniently defined by having N values and N-1 independent variables.
"""
values_per_entity(model, u::JutulVariables) = degrees_of_freedom_per_entity(model, u)

## Update functions
"""
Absolute allowable change for variable during a nonlinear update.
"""
absolute_increment_limit(::JutulVariables) = nothing

"""
Relative allowable change for variable during a nonlinear update.
A variable with value |x| and relative limit 0.2 cannot change more
than |x|*0.2.
"""
relative_increment_limit(::JutulVariables) = nothing

"""
Upper (inclusive) limit for variable.
"""
maximum_value(::JutulVariables) = nothing

"""
Lower (inclusive) limit for variable.
"""
minimum_value(::JutulVariables) = nothing

function update_primary_variable!(state, p::JutulVariables, state_symbol, model, dx, w)
    entity = associated_entity(p)
    active = active_entities(model.domain, entity, for_variables = true)
    v = state[state_symbol]
    update_jutul_variable_internal!(v, active, p, dx, w)
end

function update_jutul_variable_internal!(v::AbstractVector, active, p, dx, w)
    nu = length(active)
    abs_max = absolute_increment_limit(p)
    rel_max = relative_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)
    scale = variable_scale(p)
    @inbounds for i in 1:nu
        a_i = active[i]
        v[a_i] = update_value(v[a_i], w*dx[i], abs_max, rel_max, minval, maxval, scale)
    end
end

function update_jutul_variable_internal!(v::AbstractMatrix, active, p, dx, w)
    nu = length(active)
    n = size(v, 1)
    abs_max = absolute_increment_limit(p)
    rel_max = relative_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)
    scale = variable_scale(p)
    @inbounds for i in 1 : nu
        a_i = active[i]
        for j in 1 : n
            v[j, a_i] = update_value(v[j, a_i], w*dx[i + (j - 1)*nu], abs_max, rel_max, minval, maxval, scale)
        end
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
@inline limit_abs(dv, abs_change) = sign(dv)*min(abs(dv), abs_change)
@inline limit_abs(dv, ::Nothing) = dv
# Limit relative 
@inline limit_rel(v, dv, rel_change) = limit_abs(dv, rel_change*abs(v))
@inline limit_rel(v, dv, ::Nothing) = dv
# Lower bounds
@inline limit_upper(v, dv, maxval) = min(dv, maxval - v)
@inline limit_upper(v, dv, maxval::Nothing) = dv

# Upper bounds
@inline limit_lower(v, dv, minval) = max(dv, minval - v)
@inline limit_lower(v, dv, minval::Nothing) = dv

# Scaling
@inline scale_increment(dv, scale) = dv*scale
@inline scale_increment(dv, ::Nothing) = dv

@inline update_value(v, dv, arg...) = v + choose_increment(value(v), dv, arg...)
# Julia doesn't specialize on splatting, add full version just in case
@inline update_value(v, dv, abs, rel, minv, maxv, scale) = v + choose_increment(value(v), dv, abs, rel, minv, maxv, scale)

function get_names(v::JutulVariables)
    return (get_name(v))
end

function get_symbol(v::JutulVariables)
    return Symbol(typeof(v))
end

function get_name(v::JutulVariables)
    return String(get_symbol(v))
end

replace_value(v, new_v) = v - value(v) + new_v

"""
Define a "typical" numerical value for a variable to scale the linear system entries.
"""
variable_scale(::JutulVariables) = nothing

## Initialization
function initialize_primary_variable_ad!(state, model, pvar::ScalarVariable, state_symbol, npartials; offset = 0, kwarg...)
    diag_value = variable_scale(pvar)
    if isnothing(diag_value)
        diag_value = 1.0
    end
    initialize_variable_ad!(state, model, pvar, state_symbol, npartials, offset + 1; diag_value = diag_value, kwarg...)
end

function initialize_primary_variable_ad!(state, model, pvar, state_symbol, npartials; offset = 0, kwarg...)
    diag_value = variable_scale(pvar)
    if isnothing(diag_value)
        diag_value = 1.0
    end
    N = values_per_entity(model, pvar)
    dp = (offset+1):(offset+N)
    initialize_variable_ad!(state, model, pvar, state_symbol, npartials, offset + 1; diag_pos = dp, diag_value = diag_value, kwarg...)
end

function initialize_secondary_variable_ad!(state, model, pvar, state_symbol, npartials; kwarg...)
    diag_pos = NaN
    initialize_variable_ad!(state, model, pvar, state_symbol, npartials, diag_pos; kwarg...)
end

function initialize_variable_ad!(state, model, pvar, symb, npartials, diag_pos; kwarg...)
    state[symb] = allocate_array_ad(state[symb], diag_pos = diag_pos, context = model.context, npartials = npartials; kwarg...)
    return state
end

function initialize_variable_value(model, pvar, val; perform_copy = true)
    nu = number_of_entities(model, pvar)
    nv = values_per_entity(model, pvar)
    
    if isa(pvar, ScalarVariable)
        if val isa AbstractVector
            @assert length(val) == nu "Expected $nu entries, but got $(length(val)) for $(typeof(pvar))"
        else
            val = repeat([val], nu)
        end
        # Type-assert that this should be scalar, with a vector input
        val::AbstractVector
    else
        if isa(val, Real)
            val = repeat([val], nv, nu)
        end
        err_str = "Passed value for $(typeof(pvar))"
        nm = length(val)
        @assert nm == nv*nu "$err_str had $nm entries, expected $(nu*nv)"
        n, m, = size(val)
        @assert n == nv "$err_str had $n rows, expected $nv"
        @assert m == nu "$err_str had $m rows, expected $nu"
    end
    if perform_copy
        val = deepcopy(val)
    end
    if eltype(val)<:Real
        minv = minimum_value(pvar)
        if isnothing(minv)
            minv = -Inf
        end
        maxv = maximum_value(pvar)
        if isnothing(maxv)
            maxv = Inf
        end
        clamp!(val, minv, maxv)
    end
    return transfer(model.context, val)
end

default_value(model, variable) = 0.0
default_values(model, var::ScalarVariable) = repeat([default_value(model, var)], number_of_entities(model, var))
default_values(model, var::JutulVariables) = repeat([default_value(model, var)], values_per_entity(model, var), number_of_entities(model, var))

need_default_primary(model, var) = true

function initialize_variable_value!(state, model, pvar, symb, val; kwarg...)
    state[symb] = initialize_variable_value(model, pvar, val; kwarg...)
    return state
end

function initialize_variable_value!(state, model, pvar, symb, val::AbstractDict; need_value = true)
    if haskey(val, symb)
        value = val[symb]
    elseif need_value && need_default_primary(model, pvar)
        k = keys(val)
        error("The key $symb must be present to initialize the state. Provided symbols in initialization Dict: $k")
    else
        # We do not really need to initialize this, as it will be updated elsewhere.
        value = default_values(model, pvar)
    end
    return initialize_variable_value!(state, model, pvar, symb, value)
end

# Scalar primary variables
function initialize_variable_value(model, pvar::ScalarVariable, val::Number)
    V = repeat([val], number_of_entities(model, pvar))
    return initialize_variable_value(model, pvar, V)
end

function initialize_parameter_value!(parameters, data_domain, model, param, symb, initializer::AbstractDict)
    if haskey(initializer, symb)
        vals = initializer[symb]
    else
        vals = default_parameter_values(data_domain, model, param, symb)
    end
    return initialize_variable_value!(parameters, model, param, symb, vals)
end

function default_parameter_values(data_domain, model, param, symb)
    return default_values(model, param)
end

"""
Initializer for the value of non-scalar primary variables
"""
function initialize_variable_value(model, pvar::VectorVariables, val::AbstractVector)
    n = values_per_entity(model, pvar)
    m = number_of_entities(model, pvar)
    nv = length(val)
    if nv == n
        # One vector that should be repeated for all entities
        V = repeat(val, 1, number_of_entities(model, pvar))
    elseif nv == m
        # Vector with one entry for each cell - convert to matrix.
        V = repeat(val, 1, 1)
    else
        error("Variable $(typeof(pvar)) should have initializer of length $n or $m")
    end
    return initialize_variable_value(model, pvar, V)
end

function initialize_variable_value(model, pvar::VectorVariables, symb::Symbol, val::Number)
    n = values_per_entity(model, pvar)
    return initialize_variable_value(model, pvar, symb, repeat([val], n))
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

function update_primary_variable!(state, p::FractionVariables, state_symbol, model, dx, w)
    s = state[state_symbol]
    unit_sum_update!(s, p, model, dx, w)
end

function unit_sum_update!(s, p, model, dx, w, entity = Cells())
    nf, nu = value_dim(model, p)
    abs_max = absolute_increment_limit(p)
    maxval, minval = maximum_value(p), minimum_value(p)
    active_cells = active_entities(model.domain, entity, for_variables = true)
    if nf == 2
        unit_update_pairs!(s, dx, active_cells, minval, maxval, abs_max, w)
    else
        if true
            # Preserve direction
            unit_update_direction!(s, dx, nf, nu, active_cells, minval, maxval, abs_max, w)
        else
            # Preserve update magnitude
            unit_update_magnitude!(s, dx, nf, nu, active_cells, minval, maxval, abs_max)
        end
    end
end

function unit_update_direction!(s, dx, nf, nu, active_cells, minval, maxval, abs_max, w)
    nactive = length(active_cells)
    for active_ix in eachindex(active_cells)
        full_cell = active_cells[active_ix]
        unit_update_direction_local!(s, active_ix, full_cell, dx, nf, nactive, minval, maxval, abs_max, w)
    end
end

function unit_update_direction_local!(s, active_ix, full_cell, dx, nf, nu, minval, maxval, abs_max, w0)
    # active_ix: Index into active cells (used to access dx)
    # full_cell: Index into full set of cells (used to access s)
    w = 1.0
    # First pass: Find the relaxation factors that keep all fractions in [0, 1]
    # and obeying the maximum change targets
    dlast0 = 0.0
    @inbounds for i = 1:(nf-1)
        v = value(s[i, full_cell])
        dv0 = dx[active_ix + (i-1)*nu]
        dv = choose_increment(v, dv0, abs_max, nothing, minval, maxval)
        dlast0 -= dv0
        w = pick_relaxation(w, dv, dv0)
    end
    # Do the same thing for the implicit update of the last value
    dlast = choose_increment(value(s[nf, full_cell]), dlast0, abs_max, nothing, minval, maxval)
    w = w0*pick_relaxation(w, dlast, dlast0)

    bad_update = w <= MINIMUM_SAT_RELAX
    if bad_update
        w = w0
    end
    @inbounds for i in 1:(nf-1)
        s[i, full_cell] += w*dx[active_ix + (i-1)*nu]
    end
    @inbounds s[nf, full_cell] += w*dlast0
    if bad_update
        # Dampening is tiny, update and renormalize instead
        tot = 0.0
        @inbounds for i in 1:nf
            s_i = s[i, full_cell]
            sat = clamp(value(s_i), minval, maxval)
            tot += sat
            s_i = replace_value(s_i, sat)
            s[i, full_cell] = s_i
        end
        @inbounds for i = 1:nf
            s_i = s[i, full_cell]
            s_i = replace_value(s_i, value(s_i)/tot)
            s[i, full_cell] = s_i
        end
    end
end

function unit_update_pairs!(s, dx, active_cells, minval, maxval, abs_max, w)
    F = eltype(dx)
    maxval = min(1 - minval, maxval)
    minval = max(minval, maxval - 1)
    @inbounds for (i, cell) in enumerate(active_cells)
        v = value(s[1, cell])::F
        dv = dx[i]
        dv = w*choose_increment(v, dv, abs_max, nothing, minval, maxval)
        s[1, cell] += dv
        s[2, cell] -= dv
    end
end

function unit_update_magnitude!(s, dx, nf, nu, active_cells, minval, maxval, abs_max)
    for cell in active_cells
        unit_update_magnitude_local!(s, cell, dx, nf, nu, minval, maxval, abs_max)
    end
end

function unit_update_magnitude_local!(s, cell, dx, nf, nu, minval, maxval, abs_max)
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
    return min(w, MINIMUM_SAT_RELAX)
end
