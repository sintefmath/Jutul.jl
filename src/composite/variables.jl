function update_secondary_variables_state!(state, model::CompositeModel, vars = model.secondary_variables)
    models = model.extra[:models]
    # TODO: This is a bit hacky
    for s in keys(vars)
        name_and_var = model.secondary_variables[s]
        name, var = name_and_var
        v = state[s]
        ix = entity_eachindex(v)
        m = models[name]
        update_secondary_variable!(v, var, m, state, ix)
    end
end

function update_primary_variable!(state, p::Pair{Symbol, V}, state_symbol, model::CompositeModel, dx, w) where V<:JutulVariables
    label, var = p
    m = composite_submodel(model, label)
    update_primary_variable!(state, var, state_symbol, m, dx, w)
end
