function update_model_pre_selection!(model::CompositeModel)
    models = Dict{Symbol, JutulModel}()
    for k in keys(model.system.systems)
        models[k] = generate_submodel(model, k)
    end
    model.extra[:models] = models
    return model
end

function update_model_post_selection!(model::CompositeModel)
    pvars = get_primary_variables(model)
    for (k, m) in model.extra[:models]
        pvars_k = m.primary_variables
        empty!(pvars_k)
        for (pkey, pvar) in pvars
            pvars_k[pkey] = pvar[2]
        end
    end
    return model
end

function submodel(model::CompositeModel, k::Symbol)
    return model.extra[:models][k]
end

function default_values(model::CompositeModel, u::Pair{Symbol, V}) where V<:JutulVariables
    default_values(submodel(model, u[1]), u[2])
end

function initialize_variable_value(model::CompositeModel, pvar::Pair{Symbol, V}, val; kwarg...) where V<:JutulVariables
    m = submodel(model, pvar[1])
    initialize_variable_value(m, pvar[2], val; kwarg...)
end

function initialize_variable_ad!(state, model::CompositeModel, pvar::Pair{Symbol, V}, symb, npartials, diag_pos; kwarg...) where V<:JutulVariables
    state[symb] = allocate_array_ad(state[symb], diag_pos = diag_pos, context = model.context, npartials = npartials; kwarg...)
    return state
end

function number_of_entities(model::CompositeModel, u::Pair{Symbol, V}) where V<:JutulVariables
    number_of_entities(submodel(model, u[1]), u[2])
end

function associated_entity(u::Pair{Symbol, <:Any})
    associated_entity(u[2])
end

function variable_scale(u::Pair{Symbol, V}) where V<:JutulVariables
    variable_scale(u[2])
end

function values_per_entity(model::CompositeModel, u::Pair{Symbol, V}) where V<:JutulVariables
    values_per_entity(submodel(model, u[1]), u[2])
end

function degrees_of_freedom_per_entity(model::CompositeModel, u::Pair{Symbol, V}) where V<:JutulVariables
    degrees_of_freedom_per_entity(submodel(model, u[1]), u[2])
end

function number_of_degrees_of_freedom(model::CompositeModel, u::Pair{Symbol, V}) where V<:JutulVariables
    number_of_degrees_of_freedom(submodel(model, u[1]), u[2])
end

function initialize_primary_variable_ad!(stateAD, model, pvar::Pair{Symbol, V}, pkey, n_partials; kwarg...) where V<:JutulVariables
    m = submodel(model, pvar[1])
    return initialize_primary_variable_ad!(stateAD, m, pvar[2], pkey, n_partials; kwarg...)
end

function declare_sparsity(model::CompositeModel, eq::Pair{Symbol, V}, eq_s, u, row_layout, col_layout) where V<:JutulEquation
    k, eq = eq
    return declare_sparsity(submodel(model, k), eq, eq_s, u, row_layout, col_layout)
end

function number_of_equations(model::CompositeModel, eq::Pair{Symbol, V}) where V<:JutulEquation
    k, eq = eq
    return number_of_equations(submodel(model, k), eq)
end

function number_of_equations_per_entity(model::CompositeModel, eq::Pair{Symbol, V}) where V<:JutulEquation
    k, eq = eq
    return number_of_equations_per_entity(submodel(model, k), eq)
end

function number_of_entities(model::CompositeModel, eq::Pair{Symbol, V}) where V<:JutulEquation
    k, eq = eq
    return number_of_entities(submodel(model, k), eq)
end

function align_to_jacobian!(eq_s, eqn::Pair{Symbol, V}, jac, model::CompositeModel, u; kwarg...) where V<:JutulEquation
    k, eq = eqn
    return align_to_jacobian!(eq_s, eq, jac, submodel(model, k), u; kwarg...)
end

function update_equation!(eq_s, eqn::Pair{Symbol, V}, storage, model::CompositeModel, dt) where V<:JutulEquation
    k, eq = eqn
    return update_equation!(eq_s, eq, storage, submodel(model, k), dt)
end

function update_equation_in_entity!(eq_buf, c, state, state0, eqn::Pair{Symbol, V}, model::CompositeModel, dt, ldisc = nothing) where V<:JutulEquation
    k, eq = eqn
    if isnothing(ldisc)
        ldisc = local_discretization(eq, c)
    end
    return update_equation_in_entity!(eq_buf, c, state, state0, eq, submodel(model, k), dt, ldisc)
end

function setup_equation_storage(model::CompositeModel, eqn::Pair{Symbol, V}, storage; kwarg...) where V<:JutulEquation
    k, eq = eqn
    return setup_equation_storage(submodel(model, k), eq, storage; kwarg...)
end

function update_linearized_system_equation!(nz, r, model::CompositeModel, eqn::Pair{Symbol, V}, storage) where V<:JutulEquation
    k, eq = eqn
    return update_linearized_system_equation!(nz, r, submodel(model, k), eq, storage)
end

function convergence_criterion(model::CompositeModel, storage, eqn::Pair{Symbol, V}, eq_s, r; kwarg...) where V<:JutulEquation
    k, eq = eqn
    return convergence_criterion(submodel(model, k), storage, eq, eq_s, r; kwarg...)
end
