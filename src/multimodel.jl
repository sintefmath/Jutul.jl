export MultiModel

struct MultiModel <: TervModel
    models::NamedTuple
end

function number_of_models(model::MultiModel)
    return length(model.models)
end

function get_primary_variable_names(model::MultiModel)

end


function setup_state!(state, model::MultiModel, init_values)
    error("Mutating version of setup_state not supported for multimodel.")
end

function initialize_storage!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, initialize_storage!)
end

function allocate_storage!(storage, model::MultiModel)
    for key in keys(model.models)
        storage[key] = allocate_storage(model.models[key])
    end
end

function allocate_equations!(storage, model::MultiModel, lsys, npartials)
    @assert false "Needs implementation"
end

function allocate_linearized_system!(storage, model::TervModel)
    @assert false "Needs implementation"
end

function update_equations!(storage, model::MultiModel, arg...)
    # Might need to update this part
    submodels_storage_apply!(storage, model, update_linearized_system!, arg...)
end

function update_linearized_system!(storage, model::MultiModel, arg...)
    submodels_storage_apply!(storage, model, update_linearized_system!, arg...)
end

function setup_state(model::MultiModel, subs...)
    @assert length(subs) == number_of_models(model)
    state = Dict()
    for (i, key) in enumerate(keys(model.models))
        m = model.models[key]
        state[key] = setup_state(m, subs[i])
    end
    return state
end

function setup_parameters(model::MultiModel)
    p = Dict()
    for key in keys(model.models)
        m = model.models[key]
        p[key] = setup_parameters(m)
    end
    return p
end

function convert_state_ad(model::MultiModel, state)
    for key in keys(model.models)
        state[key] = convert_state_ad(model.models[key], state[key])
    end
end

function update_properties!(storage, model)
    submodels_storage_apply!(storage, model, update_properties!)
end

function update_equations!(storage, model, dt)
    submodels_storage_apply!(storage, model, update_equations!, dt)
end

function apply_forces!(storage, model, forces)
    for key in keys(model.models)
        apply_forces!(storage[key], model.models[key], forces[key])
    end
end


function submodels_storage_apply!(storage, model, f!, arg...)
    for key in keys(model.models)
        @show key
        @show storage
        f!(storage[key], model.models[key], arg...)
    end
end
