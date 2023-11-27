struct ScalarizedJutulVariables{T}
    vals::T
end

Base.getindex(v::ScalarizedJutulVariables, i) = Base.getindex(v.vals, i)
Base.length(v::ScalarizedJutulVariables) = Base.length(v.vals)

function Base.zero(v::ScalarizedJutulVariables{T}) where T
    return ScalarizedJutulVariables{T}(map(zero, v.vals))
end

function Base.zero(::Type{ScalarizedJutulVariables{T}}) where T
    t = tuple(T.parameters...)
    return ScalarizedJutulVariables{T}(map(zero, t))
end

"""
    scalarized_primary_variable_type(model, var::Jutul.ScalarVariable)

Get the type of a scalarized numerical variable (=Float64 for variables that are
already represented as scalars)
"""
function scalarized_primary_variable_type(model, var::Jutul.ScalarVariable)
    return Float64
end

function scalarized_primary_variable_type(model, vars::Tuple)
    types = map(x -> scalarized_primary_variable_type(model, x), vars)
    return ScalarizedJutulVariables{Tuple{types...}}
end

function scalarized_primary_variable_type(model, var::NamedTuple)
    return scalarized_primary_variable_type(model, values(var))
end

function scalarized_primary_variable_type(model, var::AbstractDict)
    return scalarized_primary_variable_type(model, tuple(values(var)...))
end

"""
    scalarize_primary_variable(model, source_vec, var::Jutul.ScalarVariable, index)

Scalarize a primary variable. For scalars, this means getting the value itself.
"""
function scalarize_primary_variable(model, source_vec, var::Jutul.ScalarVariable, index)
    return value(source_vec[index])
end

"""
    descalarize_primary_variable!(dest_array, model, V, var::Jutul.ScalarVariable, index)

Descalarize a primary variable, overwriting dest_array at entity `index`. The AD
status of entries in `dest_array` will be retained.
"""
function descalarize_primary_variable!(dest_array, model, V, var::Jutul.ScalarVariable, index)
    dest_array[index] = Jutul.replace_value(dest_array[index], V)
end

function scalarized_primary_variable_type(model, var::Jutul.FractionVariables)
    n = degrees_of_freedom_per_entity(model, var)
    if n == 1
        T = Float64
    else
        T = NTuple{n, Float64}
    end
    return T
end

function scalarize_primary_variable(model, source_mat, var::Jutul.FractionVariables, index)
    n = degrees_of_freedom_per_entity(model, var)
    if n == 1
        return value(source_mat[1, index])
    else
        ix = tuple((1:n-1)...)
        return map(i -> value(source_mat[i, index]), ix)
    end
end

function descalarize_primary_variable!(dest_array, model, V, var::Jutul.FractionVariables, index)
    rem = Jutul.maximum_value(var) - sum(V)
    for i in eachindex(V)
        dest_array[i, index] = Jutul.replace_value(dest_array[i, index], V[i])
    end
    dest_array[end, index] = Jutul.replace_value(dest_array[end, index], rem)
end

"""
    scalarize_primary_variables(model, state, pvars = model.primary_variables)

Create a vector where each entry corresponds to a tuple of values that minimally
defines the given variables. All variables must belong to the same type of
entity. This is checked by this function.
"""
function scalarize_primary_variables(model, state, pvars = model.primary_variables)
    e = missing
    pvars = (; pairs(pvars)...)
    for (k, pvar) in pairs(pvars)
        e_pvar = Jutul.associated_entity(pvar)
        if ismissing(e)
            e = e_pvar
        else
            @assert e == e_pvar
        end
    end
    T = scalarized_primary_variable_type(model, pvars)
    ne = Jutul.count_active_entities(model.domain, e)
    V = Vector{T}(undef, ne)
    return scalarize_primary_variables!(V, model, state, pvars)
end

"""
    scalarize_primary_variables!(V::Vector{T}, model, state, pvars::NamedTuple) where T

Scalarize into array. See [scalarize_primary_variables](@ref) for more details.
"""
function scalarize_primary_variables!(V::Vector{ScalarizedJutulVariables{T}}, model, state, pvars::NamedTuple) where T
    pvars_def = values(pvars)
    pvars_keys = keys(pvars)
    for i in eachindex(V)
        val = map((k, d) -> scalarize_primary_variable(model, state[k], d, i), pvars_keys, pvars_def)
        V[i] = ScalarizedJutulVariables(val)
    end
    V
end

"""
    descalarize_primary_variables!(state, model, V, pvars::NamedTuple = (; pairs(model.primary_variables)...), ind = eachindex(V))

Replace valeus in `state` by the scalarized values found in V.
"""
function descalarize_primary_variables!(state, model, V, pvars::NamedTuple = (; pairs(model.primary_variables)...), ind = eachindex(V))
    pvars_def = values(pvars)
    pvars_keys = keys(pvars)
    for (j, k, pvar) in zip(eachindex(pvars_def), pvars_keys, pvars_def)
        vals = state[k]
        for i in ind
            descalarize_primary_variable!(vals, model, V[i][j], pvar, i)
        end
    end
    return state
end
