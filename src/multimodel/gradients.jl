
function swap_primary_with_parameters!(pmodel::MultiModel, model::MultiModel)
    for k in submodel_symbols(pmodel)
        swap_primary_with_parameters!(pmodel.models[k], model.models[k])
        @info k
        display(pmodel.models[k])
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
