export state_gradient
function state_gradient(model, state, F, extra_arg...; parameters = setup_parameters(model))
    # Either with respect to all primary variables, or all parameters.
    tag = nothing
    state = merge(state, parameters)
    state = convert_state_ad(model, state, tag)
    state = convert_to_immutable_storage(state)
    n_total = number_of_degrees_of_freedom(model)

    ∂F∂x = zeros(n_total)
    state_gradient_inner!(∂F∂x, F, model, state, tag, extra_arg)
    return ∂F∂x
end

function state_gradient_inner!(∂F∂x, F, model, state, tag, extra_arg)
    layout = matrix_layout(model.context)
    function diff_entity!(∂F∂x, state, i, S, ne, np, offset)
        state_i = local_ad(state, i, S)
        v = F(model, state_i, extra_arg...)
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

function solve_adjoint(model, states, reports, G; forces = setup_forces(model), state0 = setup_state(model), parameters = setup_parameters(model))
    # One simulator object for the equations with respect to primary (at previous time-step)
    # One simulator object for the equations with respect to parameters
    # For model equations F the gradient with respect to parameters p is
    # ∇ₚG = Σₙ ∂Fₙ / ∂p λₙ where n ∈ [1, N]
    # Given Lagrange multipliers λₙ from the adjoint equations
    # (∂Fₙᵀ / ∂xₙ) λₙ = - ∂Jᵀ / ∂xₙ - (∂Fₙ₊₁ᵀ / ∂xₙ) λₙ₊₁
    # where the last term is omitted for step n = N and G is the objective function
    primary_model = adjoint_model_copy(model)
    forward_sim = Simulator(primary_model, state0 = copy(state0), parameters = copy(parameters), adjoint = true)
    backward_sim = Simulator(primary_model, state0 = copy(state0), parameters = copy(parameters), adjoint = true)

    parameter_model = adjoint_parameter_model(model)
    parameter_sim = Simulator(parameter_model, state0 = copy(parameters), parameters = copy(state0), adjoint = true)

    timesteps = report_timesteps(reports)
    N = length(states)
    @assert length(reports) == N == length(timesteps)

    n_pvar = number_of_degrees_of_freedom(model)
    n_param = number_of_degrees_of_freedom(parameter_model)
    @info "Solving adjoint for $N steps with $n_pvar primary variables and $n_param parameters."
    ∇G = zeros(n_param)

    for i in N:-1:1
        update_sensitivities!(∇G, i, G, forward_sim, backward_sim, parameter_sim, states, timesteps, forces)
    end
    return ∇G
end

function update_sensitivities!(out, i, G, forward_sim, backward_sim, parameter_sim, states, timesteps, all_forces)
    N = length(timesteps)
    forces = forces_for_timestep(forward_sim, all_forces, timesteps, i)
    dt = timesteps[i]
    state = states[i]
    dJdx = state_gradient(forward_sim.model, state, G, dt, i, forces)

    if i < N
        forces_next = forces_for_timestep(forward_sim, all_forces, timesteps, i+1)
        state_next = states[i+1]
    end

    error()
end

function adjoint_parameter_model(model)
    pmodel = adjoint_model_copy(model)
    # Swap parameters and primary variables
    set_parameters!(pmodel; pairs(model.primary_variables)...)
    # Original model holds the parameters, use those
    set_primary_variables!(pmodel; pairs(model.parameters)...)
    return pmodel
end

function adjoint_model_copy(model::SimulationModel{O, S, C, F}) where {O, S, C, F}
    pvar = copy(model.primary_variables)
    svar = copy(model.secondary_variables)
    outputs = copy(model.output_variables)
    prm = copy(model.parameters)
    eqs = model.equations
    # Transpose the system
    new_context = adjoint(model.context)
    return SimulationModel{O, S, C, F}(model.domain, model.system, new_context, model.formulation, pvar, svar, prm, eqs, outputs)
end
