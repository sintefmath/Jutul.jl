export vectorize_variables, vectorize_variables!, devectorize_variables!


function vectorize_variables(model, state_or_prm, type_or_map = :primary)
    mapper = get_mapper_internal(model, type_or_map)
    n = sum(x -> x.n, values(mapper), init = 0)
    V = zeros(n)
    return vectorize_variables!(V, model, state_or_prm, mapper)
end

function vectorize_variables!(V, model, state_or_prm, type_or_map = :primary)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, v) in mapper
        (; n, offset) = v
        state_val = state_or_prm[k]
        @assert length(state_val) == n "Expected field $k to have length $n, was $(length(state_val))"
        for i in 1:n
            V[offset+i] = state_val[i]
        end
    end
    return V
end

function devectorize_variables!(state_or_prm, model, V, type_or_map = :primary)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, v) in mapper
        state_val = state_or_prm[k]
        (; n, offset) = v
        @assert length(state_val) == n
        for i in 1:n
            state_val[i] = V[offset+i]
        end
    end
    return state_or_prm
end

get_mapper_internal(model, type_or_map::Symbol) = first(variable_mapper(model, type_or_map))
get_mapper_internal(model, type_or_map) = type_or_map
