function vectorized_length(::MultiModel, mapper)
    n = 0
    for v in values(mapper)
        n += sum(x -> x.n_x, values(v), init = 0)
    end
    return n
end

function vectorize_variables!(V, model::MultiModel, state_or_prm, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, submodel) in pairs(model.models)
        if isnothing(config)
            c = nothing
        else
            c = config[k]
        end
        vectorize_variables!(V, submodel, state_or_prm[k], mapper[k], config = c)
    end
    return V
end

function devectorize_variables!(state_or_prm, model::MultiModel, V, type_or_map = :primary; config = nothing)
    mapper = get_mapper_internal(model, type_or_map)
    for (k, submodel) in pairs(model.models)
        if isnothing(config)
            c = nothing
        else
            c = config[k]
        end
        devectorize_variables!(state_or_prm[k], submodel, V, mapper[k], config = c)
    end
    return state_or_prm
end
