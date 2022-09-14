export vectorize_variables, vectorize_variables!, devectorize_variables!


function vectorize_variables(model, state_or_prm, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    n = sum(x -> x.n, values(mapper), init = 0)
    V = zeros(n)
    return vectorize_variables!(V, model, state_or_prm, mapper, config = config)
end

function vectorize_variables!(V, model, state_or_prm, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, v) in mapper
        vectorize_variable!(V, state_or_prm, k, v)
    end
    return V
end

function vectorize_variable!(V, state, k, info)
    (; n, offset) = info
    state_val = state[k]
    @assert length(state_val) == n "Expected field $k to have length $n, was $(length(state_val))"
    for i in 1:n
        V[offset+i] = state_val[i]
    end
end

function devectorize_variables!(state_or_prm, model, V, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, v) in mapper
        devectorize_variable!(state_or_prm, V, k, v)
    end
    return state_or_prm
end

function devectorize_variable!(state, V, k, info)
    (; n, offset) = info
    state_val = state[k]
    @assert length(state_val) == n "Expected field $k to have length $n, was $(length(state_val))"
    for i in 1:n
        state_val[i] = V[offset+i]
    end
end

get_mapper_internal(model, type_or_map::Symbol) = first(variable_mapper(model, type_or_map))
get_mapper_internal(model, type_or_map) = type_or_map
