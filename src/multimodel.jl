export MultiModel

struct MultiModel <: TervModel
    models::NamedTuple
end

abstract type CrossModelTerm end

"""
A cross model term where the dependency is injective and the term is additive:
(each addition to a unit in the target only depends one unit from the source, 
and is added into that position upon application)
"""
struct InjectiveCrossModelTerm
    impacted_units
    crossterm
    cross_jac_pos
    function InjectiveCrossModelTerm(target_eq, target_model, source_model)
        context = target_model.context
        overlap = get_domain_intersection(domain_unit(target_eq), target_model, source_model)

        I = index_type(context)
        # Infer Unit from target_eq
        equations_per_unit = number_of_equations_per_unit(target_eq)
        partials_per_unit = number_of_partials_per_unit(target_eq)
        noverlap = length(overlap)

        c_term = allocate_array_ad(equations_per_unit, noverlap, context = context, npartials = partials_per_unit)
        jac_pos = zeros(I, equations_per_unit*partials_per_unit, noverlap)
        # crossterm = allocate
        new(c_term, overlap, jac_pos)
    end
end

function get_domain_intersection(u::TervUnit, target_model::TervModel, source_model::TervModel)
    return get_domain_intersection(u, target_model.domain, source_model.domain)
end

function get_domain_intersection(u::TervUnit, target_d::TervDomain, source_d::TervDomain)
    nothing
end

function number_of_models(model::MultiModel)
    return length(model.models)
end

function get_primary_variable_names(model::MultiModel)

end

function setup_state!(state, model::MultiModel, init_values)
    error("Mutating version of setup_state not supported for multimodel.")
end

function setup_simulation_storage(model::MultiModel; state0 = setup_state(model), parameters = setup_parameters(model))
    storage = Dict()
    for key in keys(model.models)
        m = model.models[key]
        storage[key] = setup_simulation_storage(m, state0 = state0[key], parameters = parameters[key])
    end
    allocate_cross_model_coupling(storage, model)
    allocate_linearized_system!(storage, model)
    return storage
end

function allocate_cross_model_coupling(storage, model::MultiModel)
    crossd = Dict{Tuple{Symbol, Symbol}, Any}()
    models = model.models
    for target in keys(models)
        for source in keys(models)
            if target == source
                continue
            end
            target_model = models[target]
            source_model = models[source]
            d = Dict()
            for (key, eq) in storage[target][:equations]
                ct = InjectiveCrossModelTerm(eq, target_model, source_model)
                if length(ct.impacted_units) == 0
                    # Just insert nothing, so we can easily spot no overlap
                    ct = nothing
                end
                d[key] = ct
            end
            crossd[(target, source)] = d
        end
    end
    storage[:cross_terms] = crossd
    display(crossd)
    @assert false "Needs implementation"
end

function allocate_linearized_system!(storage, model::MultiModel)
    @assert false "Needs implementation"
end

function initialize_storage!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, initialize_storage!)
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
    stateAD = deepcopy(state)
    for key in keys(model.models)
        stateAD[key] = convert_state_ad(model.models[key], state[key])
    end
    return stateAD
end

function update_properties!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, update_properties!)
end

function update_equations!(storage, model::MultiModel, dt)
    submodels_storage_apply!(storage, model, update_equations!, dt)
end

function apply_forces!(storage, model::MultiModel, dt, forces::Dict)
    for key in keys(model.models)
        apply_forces!(storage[key], model.models[key], dt, forces[key])
    end
end

function submodels_storage_apply!(storage, model, f!, arg...)
    for key in keys(model.models)
        f!(storage[key], model.models[key], arg...)
    end
end
