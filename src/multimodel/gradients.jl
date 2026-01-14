
function swap_primary_with_parameters!(pmodel::MultiModel, model::MultiModel, targets = parameter_targets(model))
    for k in submodels_symbols(pmodel)
        swap_primary_with_parameters!(pmodel.models[k], model.models[k], targets[k])
    end
    return pmodel
end

function adjoint_model_copy(model::MultiModel; context = nothing)
    if isnothing(context)
        F = m -> adjoint_model_copy(m)
        g = model.groups
        r = model.reduction
    else
        F = m -> adjoint_model_copy(m, context = context)
        g = nothing
        r = nothing
    end
    new_models = map(F, model.models)
    ctp = copy(model.cross_terms)
    if isnothing(context)
        new_context = adjoint(model.context)
    else
        new_context = adjoint(context)
    end
    return MultiModel(new_models, context = new_context, groups = g, cross_terms = ctp, reduction = r)
end

function convert_state_ad(model::MultiModel, state, tag = nothing)
    state = copy(state)
    @assert isnothing(tag)
    for (k, m) in pairs(model.models)
        state[k] = convert_state_ad(m, state[k], k)
    end
    return state
end

function merge_state_with_parameters(model::MultiModel, state, parameters)
    for k in submodels_symbols(model)
        merge_state_with_parameters(model[k], state[k], parameters[k])
    end
    return state
end

function state_gradient_outer!(∂F∂x, obj_eval, model::MultiModel, state; sparsity = nothing)
    offset = 0
    has_sparsity = !isnothing(sparsity)
    if has_sparsity
        @. ∂F∂x = 0
    end
    local_view(F::AbstractVector, offset, n) = view(F, (offset+1):(offset+n))
    local_view(F::AbstractMatrix, offset, n) = view(F, :, (offset+1):(offset+n))

    for k in submodels_symbols(model)
        m = model[k]
        n = number_of_degrees_of_freedom(m)
        ∂F∂x_k = local_view(∂F∂x, offset, n)
        if has_sparsity
            S = sparsity[k]
        else
            S = nothing
        end
        localtag = submodel_ad_tag(model, k)
        state_gradient_inner!(∂F∂x_k, obj_eval, m, state, localtag, model, sparsity = S, symbol = k)
        offset += n
    end
    return ∂F∂x
end

function store_sensitivities(model::MultiModel, result, prm_map)
    out = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        out[k] = Dict{Symbol, Any}()
    end
    return store_sensitivities!(out, model, result, prm_map)
end

function store_sensitivities!(out, model::MultiModel, result, prm_map)
    for k in submodels_symbols(model)
        m = model[k]
        store_sensitivities!(out[k], m, result, prm_map[k])
    end
    return out
end

function get_parameter_pair(model::MultiModel, parameters, target)
    t_outer, t_inner = target
    return get_parameter_pair(model[t_outer], parameters[t_outer], t_inner)
end

function perturb_parameter!(model::MultiModel, param_i, target, i, j, sz, ϵ)
    t_outer, t_inner = target
    perturb_parameter!(model[t_outer], param_i[t_outer], t_inner, i, j, sz, ϵ)
end

function parameter_targets(model::MultiModel)
    targets = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        targets[k] = parameter_targets(model[k])
    end
    return targets
end

function variable_mapper(model::MultiModel, arg...;
        targets = nothing,
        config = nothing,
        offset_x = 0,
        offset_full = offset_x
    )
    out = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        if isnothing(targets) || !haskey(targets, k)
            t = nothing
        else
            t = targets[k]
        end
        if isnothing(config)
            c = nothing
        else
            c = config[k]
        end
        out[k], offset_full, offset_x = variable_mapper(model[k], arg...;
            targets = t,
            config = c,
            offset_full = offset_full,
            offset_x = offset_x
        )
    end
    return (out, offset_full, offset_x)
end

function rescale_sensitivities!(dG, model::MultiModel, parameter_map; renum = nothing)
    for k in submodels_symbols(model)
        rescale_sensitivities!(dG, model[k], parameter_map[k], renum = renum)
    end
end

function optimization_config(model::MultiModel, param, active = nothing; kwarg...)
    out = Dict{Symbol, OptimizationConfig}()
    for k in submodels_symbols(model)
        m = model[k]
        if isnothing(active) || !haskey(active, k)
            v = optimization_config(m, param[k]; kwarg...)
        else
            v = optimization_config(m, param[k], active[k]; kwarg...)
        end
        out[k] = v
    end
    return out
end

function optimization_targets(config::Dict, model::MultiModel)
    out = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        out[k] = optimization_targets(config[k], model[k])
    end
    return out
end

function optimization_limits!(lims, config, mapper, param, model::MultiModel)
    for k in submodels_symbols(model)
        optimization_limits!(lims, config[k], mapper[k], param[k], model[k])
    end
    return lims
end

function transfer_gradient!(dFdy, dFdx, x, mapper, config, model::MultiModel)
    for k in submodels_symbols(model)
        transfer_gradient!(dFdy, dFdx, x, mapper[k], config[k], model[k])
    end
    return dFdy
end

function swap_variables(state, parameters, model::MultiModel; kwarg...)
    out = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        out[k] = swap_variables(state[k], parameters[k], model[k]; kwarg...)
    end
    return out
end

function print_parameter_optimization_config(targets, config, model::MultiModel)
    for (k, v) in targets
        print_parameter_optimization_config(v, config[k], model[k], title = k)
    end
end

objective_depends_on_crossterms(F) = false

function determine_sparsity_simple(F, model::MultiModel, state, state0 = nothing; variant = missing)
    @assert isnothing(state0) "Not implemented."
    outer_sparsity = Dict()
    for mod_k in submodels_symbols(model)
        model_k = model[mod_k]
        if ismissing(variant)
            subkeys = nothing
        else
            if variant == :parameters
                subkeys = keys(get_parameters(model_k))
            else
                variant == :variables || error("variant must be :parameters or :variables, was $variant")
                subkeys = keys(get_variables(model_k))
            end
        end
        sparsity = Dict()
        substate = state[mod_k]
        entities = ad_entities(substate)
        for (k, v) in entities
            # Create a outer state where everything is value except current focus
            outer_state = JutulStorage()
            for (statek, statev) in pairs(state)
                outer_state[statek] = as_value(statev)
            end
            outer_state[mod_k] = create_mock_state(state[mod_k], k, entities, subkeys = subkeys)
            # Apply the function
            f_ad = F(outer_state)

            V = sum(f_ad)
            if V isa AbstractFloat
                S = zeros(Int64, 0)
            else
                D = ST.deriv(V)
                S = D.nzind
            end
            sparsity[k] = S
        end
        outer_sparsity[mod_k] = sparsity
    end
    # We also add sparsity from cross terms (they can couple variables between models)
    if objective_depends_on_crossterms(F)
        for ctp in model.cross_terms
            for is_normal in (true, false)
                ct = ctp.cross_term
                if !is_normal && !has_symmetry(ct)
                    continue
                end
                if is_normal
                    modname = ctp.target
                    eqname = ctp.target_equation
                else
                    modname = ctp.source
                    eqname = ctp.source_equation
                end
                m = model[modname]
                eq = m.equations[eqname]
                if is_normal
                    ix = cross_term_entities(ct, eq, m)
                else
                    ix = cross_term_entities_source(ct, eq, m)
                end
                entityname = associated_entity(eq)
                msparsity = outer_sparsity[modname]
                if !haskey(msparsity, entityname)
                    msparsity[entityname] = Int64[]
                end
                dest = msparsity[entityname]
                for i in ix
                    push!(dest, i)
                end
                unique!(dest)
            end
        end
    end
    return outer_sparsity
end

function solve_numerical_sensitivities(model::MultiModel, states, reports, G; kwarg...)
    out = Dict()
    for mk in submodels_symbols(model)
        inner = Dict()
        for k in keys(model[mk].parameters)
            inner[k] = solve_numerical_sensitivities(model, states, reports, G, (mk, k); kwarg...)
        end
        out[mk] = inner
    end
    return out
end

function objective_state_reference_parameters(target_state, state, model::MultiModel)
    for k in submodels_symbols(model)
        target_state[k] = objective_state_reference_parameters(target_state[k], state[k], model[k])
    end
    return target_state
end

function objective_state_shallow_copy(state, model::MultiModel)
    target_state = JutulStorage()
    for k in submodels_symbols(model)
        target_state[k] = objective_state_shallow_copy(state[k], model[k])
    end
    return target_state
end
