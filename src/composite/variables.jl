function update_secondary_variables_state!(state, model::CompositeModel)
    models = model.extra[:models]
    for (s, name_and_var) in model.secondary_variables
        name, var = name_and_var
        v = state[s]
        ix = entity_eachindex(v)
        m = models[name]
        update_secondary_variable!(v, var, m, state, ix)
    end
end

function update_primary_variable!(state, p::Pair{Symbol, V}, state_symbol, model::CompositeModel, dx, w) where V<:JutulVariables
    label, var = p
    m = submodel(model, label)
    update_primary_variable!(state, var, state_symbol, m, dx, w)
end
