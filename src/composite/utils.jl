function select_extra_model_fields!(model::CompositeModel)
    models = Dict{Symbol, JutulModel}()
    for k in keys(model.system.systems)
        models[k] = generate_submodel(model, k)
    end
    model.extra[:models] = models
    return model
end

function submodel(model::CompositeModel, k::Symbol)
    return model.extra[:models][k]
end

function default_values(model::CompositeModel, u::Tuple{Symbol, V}) where V<:JutulVariables
    default_values(submodel(model, u[1]), u[2])
end

function initialize_variable_value(model::CompositeModel, pvar::Tuple{Symbol, V}, val; kwarg...) where V<:JutulVariables
    m = submodel(model, pvar[1])
    initialize_variable_value(m, pvar[2], val; kwarg...)
end

function initialize_variable_ad!(state, model::CompositeModel, pvar::Tuple{Symbol, V}, symb, npartials, diag_pos; kwarg...) where V<:JutulVariables
    state[symb] = allocate_array_ad(state[symb], diag_pos = diag_pos, context = model.context, npartials = npartials; kwarg...)
    return state
end

function number_of_entities(model::CompositeModel, u::Tuple{Symbol, V}) where V<:JutulVariables
    number_of_entities(submodel(model, u[1]), u[2])
end

function associated_entity(u::Tuple{Symbol, V}) where V<:JutulVariables
    associated_entity(u[2])
end

function variable_scale(u::Tuple{Symbol, V}) where V<:JutulVariables
    variable_scale(u[2])
end

function values_per_entity(model::CompositeModel, u::Tuple{Symbol, V}) where V<:JutulVariables
    degrees_of_freedom_per_entity(submodel(model, u[1]), u[2])
end

function degrees_of_freedom_per_entity(model::CompositeModel, u::Tuple{Symbol, V}) where V<:JutulVariables
    degrees_of_freedom_per_entity(submodel(model, u[1]), u[2])
end

function number_of_degrees_of_freedom(model::CompositeModel, u::Tuple{Symbol, V}) where V<:JutulVariables
    number_of_degrees_of_freedom(submodel(model, u[1]), u[2])
end

function initialize_primary_variable_ad!(stateAD, model, pvar::Tuple{Symbol, V}, pkey, n_partials; kwarg...) where V<:JutulVariables
    m = submodel(model, pvar[1])
    return initialize_primary_variable_ad!(stateAD, m, pvar[2], pkey, n_partials; kwarg...)
end

# function update_primary_variable!(state, p::Tuple{Symbol, V}, state_symbol, model::CompositeModel, dx) where V<:JutulVariables

# end