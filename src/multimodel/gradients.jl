
function swap_primary_with_parameters!(pmodel::MultiModel, model::MultiModel)
    for k in submodel_symbols(pmodel)
        swap_primary_with_parameters!(pmodel.models[k], model.models[k])
    end
    return pmodel
end

function adjoint_model_copy(model::MultiModel)
    new_models = map(adjoint_model_copy, model.models)
    g = model.groups
    r = model.reduction
    ctp = copy(model.cross_terms)
    new_context = adjoint(model.context)
    return MultiModel(new_models, context = new_context, groups = g, cross_terms = ctp, reduction = r)
end

function convert_state_ad(model::MultiModel, state, tag = nothing)
    @assert isnothing(tag)
    for (k, m) in pairs(model.models)
        @info k keys(state[k])
        state[k] = convert_state_ad(m, state[k], k)
    end
end

function merge_state_with_parameters(model::MultiModel, state, parameters)
    for k in submodel_symbols(model)
        merge_state_with_parameters(model[k], state[k], parameters[k])
    end
    return state
end
