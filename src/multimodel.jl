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
    
end

function initialize_storage!(d, model::MultiModel)
    # Do nothing
end

function allocate_storage!(d, model::MultiModel)
    
end

function allocate_equations!(d, model::MultiModel, lsys, npartials)
    
end

function update_equations!(model::MultiModel, storage)
    # Do nothing
end

function update_linearized_system!(model::MultiModel, storage)
    
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
    for (i, key) in enumerate(keys(model.models))
        m = model.models[key]
        state[key] = convert_state_ad(m, state[key])
    end
end

function newton_step(model::MultiModel, storage; dt = nothing, linsolve = nothing, forces = nothing, iteration = nan)

end