export vectorize_variables, vectorize_variables!, devectorize_variables!


function vectorize_variables(model, state_or_prm, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    n = vectorized_length(model, mapper)
    V = zeros(n)
    return vectorize_variables!(V, model, state_or_prm, mapper, config = config)
end

vectorized_length(model, mapper) = sum(x -> x.n, values(mapper), init = 0)

function vectorize_variables!(V, model, state_or_prm, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, v) in mapper
        F = opt_scaler_function(config, k, inv = false)
        vectorize_variable!(V, state_or_prm, k, v, F)
    end
    return V
end

function vectorize_variable!(V, state, k, info, F)
    (; n, offset) = info
    state_val = state[k]
    @assert length(state_val) == n "Expected field $k to have length $n, was $(length(state_val))"
    for i in 1:n
        V[offset+i] = F(state_val[i])
    end
end

function devectorize_variables!(state_or_prm, model, V, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, v) in mapper
        F = opt_scaler_function(config, k, inv = true)
        devectorize_variable!(state_or_prm, V, k, v, F)
    end
    return state_or_prm
end

function devectorize_variable!(state, V, k, info, F_inv)
    (; n, offset) = info
    state_val = state[k]
    @assert length(state_val) == n "Expected field $k to have length $n, was $(length(state_val))"
    for i in 1:n
        state_val[i] = F_inv(V[offset+i])
    end
end

get_mapper_internal(model, type_or_map::Symbol) = first(variable_mapper(model, type_or_map))
get_mapper_internal(model, type_or_map) = type_or_map
