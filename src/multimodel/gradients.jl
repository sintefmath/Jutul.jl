
function swap_primary_with_parameters!(pmodel::MultiModel, model::MultiModel, targets = parameter_targets(model))
    for k in submodel_symbols(pmodel)
        swap_primary_with_parameters!(pmodel.models[k], model.models[k], targets[k])
    end
    return pmodel
end

function adjoint_model_copy(model::MultiModel)
    new_models = map(adjoint_model_copy, model.models)
    g = model.groups
    r = model.reduction
    ctp = copy(model.cross_terms)
    if isnothing(model.context)
        new_context = adjoint(DefaultContext())
    else
        new_context = adjoint(model.context)
    end
    return MultiModel(new_models, context = new_context, groups = g, cross_terms = ctp, reduction = r)
end

function convert_state_ad(model::MultiModel, state, tag = nothing)
    @assert isnothing(tag)
    for (k, m) in pairs(model.models)
        state[k] = convert_state_ad(m, state[k], k)
    end
    return state
end

function merge_state_with_parameters(model::MultiModel, state, parameters)
    for k in submodel_symbols(model)
        merge_state_with_parameters(model[k], state[k], parameters[k])
    end
    return state
end

function state_gradient_outer!(∂F∂x, F, model::MultiModel, state, extra_arg)
    offset = 0

    local_view(F::AbstractVector, offset, n) = view(F, (offset+1):(offset+n))
    local_view(F::AbstractMatrix, offset, n) = view(F, :, (offset+1):(offset+n))

    for k in submodel_symbols(model)
        m = model[k]
        n = number_of_degrees_of_freedom(m)
        ∂F∂x_k = local_view(∂F∂x, offset, n)
        state_gradient_inner!(∂F∂x_k, F, m, state, k, extra_arg, model)
        offset += n
    end
    return ∂F∂x
end

function store_sensitivities(model::MultiModel, result, prm_map)
    out = Dict{Symbol, Any}()
    for k in submodel_symbols(model)
        m = model[k]
        out[k] = Dict{Symbol, Any}()
        store_sensitivities!(out[k], m, result, prm_map[k])
    end
    return out
end

function get_parameter_pair(model::MultiModel, parameters, target)
    t_outer, t_inner = target
    return get_parameter_pair(model[t_outer], parameters[t_outer], t_inner)
end

function perturb_parameter!(model::MultiModel, param_i, target, i, ϵ)
    t_outer, t_inner = target
    perturb_parameter!(model[t_outer], param_i[t_outer], t_inner, i, ϵ)
end

function parameter_targets(model::MultiModel)
    targets = Dict{Symbol, Any}()
    for k in submodel_symbols(model)
        targets[k] = parameter_targets(model[k])
    end
    return targets
end

function variable_mapper(model::MultiModel, arg...; targets = nothing, offset = 0)
    out = Dict{Symbol, Any}()
    for k in submodel_symbols(model)
        if isnothing(targets)
            t = nothing
        else
            t = targets[k]
        end
        out[k], offset = variable_mapper(model[k], arg..., targets = t, offset = offset)
    end
    return (out, offset)
end

function rescale_sensitivities!(dG, model::MultiModel, parameter_map)
    for k in submodel_symbols(model)
        rescale_sensitivities!(dG, model[k], parameter_map[k])
    end
end

function optimization_config(model::MultiModel, param, active = nothing)
    out = Dict{Symbol, Any}()
    for k in submodel_symbols(model)
        m = model[k]
        if isnothing(active)
            v = optimization_config(m, param[k])
        else
            v = optimization_config(m, param[k], active[k])
        end
        out[k] = v
    end
    return out
end

function optimization_targets(config::Dict, model::MultiModel)
    out = Dict{Symbol, Any}()
    for k in submodel_symbols(model)
        out[k] = optimization_targets(config[k], model[k])
    end
    return out
end

function optimization_limits!(lims, config, mapper, x0, param, model::MultiModel)
    for k in submodel_symbols(model)
        optimization_limits!(lims, config[k], mapper[k], x0, param[k], model[k])
    end
    return lims
end

function transfer_gradient!(dFdy, dFdx, x, mapper, config, model::MultiModel)
    for k in submodel_symbols(model)
        transfer_gradient!(dFdy, dFdx, x, mapper[k], config[k], model[k])
    end
    return dFdy
end
