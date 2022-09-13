function vectorize_variables(model::MultiModel, state_or_prm, type_or_map = :primary)
    mapper = get_mapper_internal(model, type_or_map)
    n = 0
    for (k, v) in mapper
        n += sum(x -> x.n, values(v), init = 0)
    end
    @info n mapper
    V = zeros(n)
    vectorize_variables!(V, model, state_or_prm, mapper)
end

function vectorize_variables!(V, model::MultiModel, state_or_prm, type_or_map = :primary)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, submodel) in pairs(model.models)
        vectorize_variables!(V, submodel, state_or_prm[k], mapper[k])
    end
    return V
end

function devectorize_variables!(state_or_prm, model::MultiModel, V, type_or_map = :primary)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, submodel) in pairs(model.models)
        devectorize_variables!(state_or_prm[k], submodel, V, mapper[k])
    end
    return state_or_prm
end
