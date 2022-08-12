export state_gradient
function state_gradient(model, state, F; parameters = setup_parameters(model))
    # Either with respect to all primary variables, or all parameters.
    tag = nothing
    state = merge(state, parameters)
    state = convert_state_ad(model, state, tag)
    state = convert_to_immutable_storage(state)
    n_total = number_of_degrees_of_freedom(model)

    ∂F∂x = zeros(n_total)
    state_gradient_inner!(∂F∂x, F, model, state, tag)
    return ∂F∂x
end

function state_gradient_inner!(∂F∂x, F, model, state, tag)
    layout = matrix_layout(model.context)
    function diff_entity!(∂F∂x, state, i, S, ne, np, offset)
        state_i = local_ad(state, i, S)
        v = F(state_i)
        for p_i in np
            ix = alignment_linear_index(i, p_i, ne, np, layout) + offset
            ∂F∂x[ix] = v.partials[p_i]
        end
    end
    offset = 0
    for e in get_primary_variable_ordered_entities(model)
        np = number_of_partials_per_entity(model, e)
        ne = count_active_entities(model.domain, e)
        ltag = get_entity_tag(tag, e)
        S = typeof(get_ad_entity_scalar(1.0, np, tag = ltag))
        for i in 1:ne
            diff_entity!(∂F∂x, state, i, S, ne, np, offset)
        end
        offset += ne*np
    end
end
