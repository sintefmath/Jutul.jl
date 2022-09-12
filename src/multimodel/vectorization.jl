function vectorize_variables(model::MultiModel, state_or_prm, type = :primary)
    n = sum(m -> number_of_values(m, type), values(model.models))
    V = zeros(n)
    vectorize_variables!(V, model, state_or_prm, type)
end

function vectorize_variables!(V, model::MultiModel, state_or_prm, type = :primary, offset = 0)
    for (k, submodel) in pairs(model.models)
        vectorize_variables!(V, submodel, state_or_prm[k], type, offset)
        offset += number_of_values(submodel)
    end
    return V
end

function devectorize_variables!(state_or_prm, model::MultiModel, V, type = :primary, offset = 0)
    for (k, submodel) in pairs(model.models)
        devectorize_variables!(state_or_prm[k], submodel, V, type, offset)
        offset += number_of_values(submodel)
    end
    return state_or_prm
end
