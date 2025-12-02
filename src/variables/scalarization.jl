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
function scalarized_primary_variable_type(model, var::Jutul.ScalarVariable, T = Float64)
    return T
end

function scalarized_primary_variable_type(model, vars::Tuple, T = Float64)
    types = map(x -> scalarized_primary_variable_type(model, x), vars, T)
    return ScalarizedJutulVariables{Tuple{types...}}
end

function scalarized_primary_variable_type(model, var::NamedTuple, T = Float64)
    return scalarized_primary_variable_type(model, values(var), T)
end

function scalarized_primary_variable_type(model, var::AbstractDict, T = Float64)
    return scalarized_primary_variable_type(model, tuple(values(var)...), T)
end

function scalarize_variable(model, source_vec, var, index; numeric::Bool = false)
    return scalarize_primary_variable(model, source_vec, var, index; numeric = numeric)
end

"""
    scalarize_primary_variable(model, source_vec, var::Jutul.ScalarVariable, index)

Scalarize a primary variable. For scalars, this means getting the value itself.
"""
function scalarize_primary_variable(model, source_vec, var::Jutul.ScalarVariable, index; numeric::Bool = false)
    return value(source_vec[index])
end

function scalarize_primary_variable(model, source_vec, var::Jutul.JutulVariables, index; numeric::Bool = false)
    num = degrees_of_freedom_per_entity(model, var)
    T = eltype(source_vec)
    T_isbits = isbitstype(T)
    if num == 1
        scalar_v = value(source_vec[1, index])
    else
        if T_isbits
            tmp = @MVector zeros(T, num)
            for i in 1:num
                tmp[i] = value(source_vec[i, index])
            end
            scalar_v = SVector{num, T}(tmp)
        else
            scalar_v = zeros(T, num)
            for i in 1:num
                scalar_v[i] = value(source_vec[i, index])
            end
        end
    end
    return scalar_v
end

function descalarize_variable!(dest_array, model, V, var, index, reference = missing; F = identity)
    return descalarize_primary_variable!(dest_array, model, V, var, index, reference, F = F)
end

"""
    descalarize_primary_variable!(dest_array, model, V, var::Jutul.ScalarVariable, index)

Descalarize a primary variable, overwriting dest_array at entity `index`. The AD
status of entries in `dest_array` will be retained.
"""
function descalarize_primary_variable!(dest_array, model, V, var::Jutul.ScalarVariable, index, ref = missing;
        F = identity
    )
    dest_array[index] = Jutul.replace_value(dest_array[index], F(V))
end

function descalarize_primary_variable!(dest_array, model, V, var::Jutul.JutulVariables, index, ref = missing;
        F = identity
    )
    @assert size(dest_array, 1) == length(V)
    for i in eachindex(V)
        dest_array[i, index] = Jutul.replace_value(dest_array[i, index], F(V[i]))
    end
end

function scalarized_primary_variable_type(model, var::Jutul.FractionVariables, T_num = Float64)
    N = degrees_of_freedom_per_entity(model, var)
    if N == 1
        T = T_num
    else
        T = SVector{N, T_num}
    end
    return T
end

function scalarize_primary_variable(model, source_mat, var::Jutul.FractionVariables, index; numeric::Bool = false)
    N = degrees_of_freedom_per_entity(model, var)
    if N == 1
        scalar_v = value(source_mat[1, index])
    else
        tmp = @MVector zeros(N)
        for i in 1:N
            tmp[i] = value(source_mat[i, index])
        end
        scalar_v = SVector{N, Float64}(tmp)
    end
    return scalar_v
end

function descalarize_primary_variable!(dest_array, model, V, var::Jutul.FractionVariables, index, ref = missing;
        F = identity
    )
    rem = Jutul.maximum_value(var) - sum(V)
    for i in eachindex(V)
        dest_array[i, index] = Jutul.replace_value(dest_array[i, index], F(V[i]))
    end
    @assert size(dest_array, 1) == length(V) + 1
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

Scalarize into array. See [`scalarize_primary_variables`](@ref) for more details.
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

Replace values in `state` by the scalarized values found in V.
"""
function descalarize_primary_variables!(state, model, V, pvars::NamedTuple = (; pairs(model.primary_variables)...), ind = eachindex(V))
    pvars_def = values(pvars)
    pvars_keys = keys(pvars)
    for (j, k, pvar) in zip(eachindex(pvars_def), pvars_keys, pvars_def)
        descalarize_primary_variable_inner!(state[k], model, V, pvar, ind, Val(j))
    end
    return state
end

function descalarize_primary_variable_inner!(vals, model, V, pvar, ind, ::Val{j}) where j
    for i in ind
        descalarize_primary_variable!(vals, model, V[i][j], pvar, i)
    end
end
