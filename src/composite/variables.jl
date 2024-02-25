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

function set_variable_internal!(vars, model::CompositeModel; kwarg...)
    for (k, v) in kwarg
        if v isa Pair
            # "Right" format for a composite model.
            delete_variable!(model, k)
            vars[k] = v
        else
            if haskey(vars, k)
                vars[k]::Pair
                oldkey, = vars[k]
            else
                oldkey = missing
            end
            if ismissing(oldkey)
                mkeys = keys(model.system.systems)
                if length(mkeys) == 1
                    oldkey = only(mkeys)
                else
                    throw(ArgumentError("$k is not present in model.\nFor a composite model, this must be specified as a Pair with key as one of $mkeys.\nExample: $k = Pair(:$(first(mkeys)), var)"))
                end
            end
            v::JutulVariables
            delete_variable!(model, k)
            vars[k] = Pair(oldkey, v)
        end
    end
end

function get_dependencies(svar::Pair, model::CompositeModel)
    k, var = svar
    models = model.extra[:models]
    return get_dependencies(var, models[k])
end
