export vectorize_variables, vectorize_variables!, devectorize_variables!


function vectorize_variables(model, state_or_prm, type = :primary)
    vars = get_variables_by_type(model, type)
    n = 0
    for (k, v) in pairs(vars)
        @info k
        n += values_per_entity(model, v)*number_of_entities(model, v)
    end
    V = zeros(n)
    return vectorize_variables!(V, model, state_or_prm, type)
end

function vectorize_variables!(V, model, state_or_prm, type = :primary, offset = 0)
    vars = get_variables_by_type(model, type)
    for k in keys(vars)
        for v_i in state_or_prm[k]
            V[offset+1] = v_i
            offset += 1
        end
    end
    return V
end

function devectorize_variables!(state_or_prm, model, V, type = :primary)
    vars = get_variables_by_type(model, type)
    for k in keys(vars)
        state_val = state_or_prm[k]
        for i in eachindex(state_val)
            state_val[i] = V[offset+1]
            offset += 1
        end
    end
    return state_or_prm
end
