# Primary variables
export degrees_of_freedom_per_entity, minimum_output_variables
export absolute_increment_limit, relative_increment_limit, maximum_value, minimum_value, update_primary_variable!, default_value, initialize_variable_value!, number_of_entities
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

function number_of_degrees_of_freedom(model::JutulModel, u::JutulUnit)
    ndof = degrees_of_freedom_per_entity(model, u)*count_active_entities(model.domain, u, for_variables = true)
    return ndof
end

function degrees_of_freedom_per_entity(model::JutulModel, u::JutulUnit)
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

function value_dim(model, pvars::JutulVariables)
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

function update_primary_variable!(state, p::JutulVariables, state_symbol, model, dx)
    entity = associated_entity(p)
    active = active_entities(model.domain, entity, for_variables = true)

    n = degrees_of_freedom_per_entity(model, p)
    nu = length(active)
    abs_max = absolute_increment_limit(p)
    rel_max = relative_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)
    scale = variable_scale(p)

    for index in 1:n
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

@inline function update_value(v, dv, arg...)
    return v + choose_increment(value(v), dv, arg...)
end

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

variable_scale(::JutulVariables) = nothing

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
        nm = length(val)
        @assert nm == nv*nu "Passed value had $nm entries, expected $(nu*nv)"
        n, m, = size(val)
        @assert n == nv "Passed value had $n rows, expected $nv"
        @assert m == nu "Passed value had $m rows, expected $nu"
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

function unit_sum_update!(s, p, model, dx, entity = Cells())
    nf, nu = value_dim(model, p)
    abs_max = absolute_increment_limit(p)
    maxval, minval = maximum_value(p), minimum_value(p)
    active_cells = active_entities(model.domain, entity, for_variables = true)
    if nf == 2
        unit_update_pairs!(s, dx, active_cells, minval, maxval, abs_max)
    else
        if true
            # Preserve direction
            unit_update_direction!(s, dx, nf, nu, active_cells, minval, maxval, abs_max)
        else
            # Preserve update magnitude
            unit_update_magnitude!(s, dx, nf, nu, active_cells, minval, maxval, abs_max)
        end
    end
end

function unit_update_direction!(s, dx, nf, nu, active_cells, minval, maxval, abs_max)
    nactive = length(active_cells)
    @batch minbatch = 1000 for active_ix in eachindex(active_cells)
        full_cell = active_cells[active_ix]
        unit_update_direction_local!(s, active_ix, full_cell, dx, nf, nactive, minval, maxval, abs_max)
    end
end

function unit_update_direction_local!(s, active_ix, full_cell, dx, nf, nu, minval, maxval, abs_max)
    # active_ix: Index into active cells (used to access dx)
    # full_cell: Index into full set of cells (used to access s)
    w = 1.0
    # First pass: Find the relaxation factors that keep all fractions in [0, 1]
    # and obeying the maximum change targets
    dlast0 = 0
    @inbounds for i = 1:(nf-1)
        v = value(s[i, full_cell])
        dv0 = dx[active_ix + (i-1)*nu]
        dv = choose_increment(v, dv0, abs_max, nothing, minval, maxval)
        dlast0 -= dv0
        w = pick_relaxation(w, dv, dv0)
    end
    # Do the same thing for the implicit update of the last value
    dlast = choose_increment(value(s[nf, full_cell]), dlast0, abs_max, nothing, minval, maxval)
    w = pick_relaxation(w, dlast, dlast0)

    @inbounds for i = 1:(nf-1)
        s[i, full_cell] += w*dx[active_ix + (i-1)*nu]
    end
    @inbounds s[nf, full_cell] += w*dlast0
end

function unit_update_pairs!(s, dx, active_cells, minval, maxval, abs_max)
    F = eltype(dx)
    maxval = min(1 - minval, maxval)
    minval = max(minval, maxval - 1)
    @inbounds for (i, cell) in enumerate(active_cells)
        v = value(s[1, cell])::F
        dv = dx[i]
        dv = choose_increment(v, dv, abs_max, nothing, minval, maxval)
        s[1, cell] += dv
        s[2, cell] -= dv
    end
end

function unit_update_magnitude!(s, dx, nf, nu, active_cells, minval, maxval, abs_max)
    @batch minbatch = 1000 for cell in active_cells
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
    return w
end

function values_per_entity(model, var::ConstantVariables)
    c = var.constants
    if var.single_entity
        n = length(c)
    elseif isa(c, AbstractVector)
        n = 1
    else
        n = size(c, 1)
    end
    return n
end
associated_entity(model, var::ConstantVariables) = var.entity

function transfer(context, v::ConstantVariables)
    constants = transfer(context, v.constants)
    return ConstantVariables(constants, v.entity, single_entity = v.single_entity)
end

update_secondary_variable!(x, var::ConstantVariables, model, parameters, state) = nothing

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

subvariable(var::ConstantVariables, map::TrivialGlobalMap) = var

function subvariable(var::ConstantVariables, map::FiniteVolumeGlobalMap)
    if var.entity == Cells()
        if !var.single_entity
            c = copy(var.constants)
            p_i = map.cells
            if isa(c, AbstractVector)
                c = c[p_i]
            else
                c = c[:, p_i]
            end
            var = ConstantVariables(c, var.entity, single_entity = false)
        end
    else
        error("Mappings other than cells not implemented")
    end
    return var
end
