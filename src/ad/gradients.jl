export state_gradient
function state_gradient(model, state, F, extra_arg...; kwarg...)
    n_total = number_of_degrees_of_freedom(model)
    ∂F∂x = zeros(n_total)
    return state_gradient!(∂F∂x, model, state, F, extra_arg...; kwarg...)
end

function state_gradient!(∂F∂x, model, state, F, extra_arg...; parameters = setup_parameters(model))
    # Either with respect to all primary variables, or all parameters.
    tag = nothing
    state = merge(state, parameters)
    state = convert_state_ad(model, state, tag)
    state = convert_to_immutable_storage(state)
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

function solve_adjoint_sensitivities(model, states, reports, G; forces = setup_forces(model), state0 = setup_state(model), parameters = setup_parameters(model))
    # One simulator object for the equations with respect to primary (at previous time-step)
    # One simulator object for the equations with respect to parameters
    # For model equations F the gradient with respect to parameters p is
    # ∇ₚG = Σₙ ∂Fₙ / ∂p λₙ where n ∈ [1, N]
    # Given Lagrange multipliers λₙ from the adjoint equations
    # (∂Fₙᵀ / ∂xₙ) λₙ = - ∂Jᵀ / ∂xₙ - (∂Fₙ₊₁ᵀ / ∂xₙ) λₙ₊₁
    # where the last term is omitted for step n = N and G is the objective function
    primary_model = adjoint_model_copy(model)
    # Standard model for: ∂Fₙᵀ / ∂xₙ
    forward_sim = Simulator(primary_model, state0 = deepcopy(state0), parameters = deepcopy(parameters), adjoint = false)
    # Same model, but adjoint for: ∂Fₙ₊₁ᵀ / ∂xₙ
    backward_sim = Simulator(primary_model, state0 = deepcopy(state0), parameters = deepcopy(parameters), adjoint = true)
    # Create parameter model for ∂Fₙ / ∂p
    parameter_model = adjoint_parameter_model(model)
    parameter_sim = Simulator(parameter_model, state0 = deepcopy(parameters), parameters = deepcopy(state0), adjoint = false)

    timesteps = report_timesteps(reports)
    N = length(states)
    @assert length(reports) == N == length(timesteps)

    n_pvar = number_of_degrees_of_freedom(model)
    n_param = number_of_degrees_of_freedom(parameter_model)
    @info "Solving adjoint for $N steps with $n_pvar primary variables and $n_param parameters."
    ∇G = zeros(n_param)
    λ = zeros(n_pvar)

    for i in N:-1:1
        if i == 1
            s0 = deepcopy(state0)
        else
            s0 = states[i-1]
        end
        if i == N
            s_next = nothing
        else
            s_next = states[i+1]
        end
        s = states[i]
        update_sensitivities!(λ, ∇G, i, G, forward_sim, backward_sim, parameter_sim, s0, s, s_next, timesteps, forces)
    end
    return ∇G
end

function update_sensitivities!(λ, ∇G, i, G, forward_sim, backward_sim, parameter_sim, state0, state, state_next, timesteps, all_forces)
    N = length(timesteps)
    forces = forces_for_timestep(forward_sim, all_forces, timesteps, i)
    dt = timesteps[i]
    # Assemble Jacobian w.r.t. current step
    adjoint_reassemble!(forward_sim, state, state0, dt, forces)
    dGdx = state_gradient(forward_sim.model, state, G, dt, i, forces)

    lsys = forward_sim.storage.LinearizedSystem
    rhs = lsys.r_buffer
    @. rhs = -dGdx
    if isnothing(state_next)
        @assert i == N
    else
        dt_next = timesteps[i+1]
        forces_next = forces_for_timestep(backward_sim, all_forces, timesteps, i+1)
        adjoint_reassemble!(backward_sim, state_next, state, dt_next, forces_next)
        lsys_next = backward_sim.storage.LinearizedSystem
        op = linear_operator(lsys_next)
        # rhs -= op*λ
        mul!(rhs, op, λ, 1.0, -1.0)
    end
    # We have the right hand side, assemble the Jacobian and solve for the Lagrange multiplier
    solve!(lsys)
    @. λ -= lsys.dx_buffer
    # ∇ₚG = Σₙ ∂Fₙ / ∂p λₙ
    # Increment gradient
    adjoint_reassemble!(parameter_sim, state, state0, dt, forces)
    lsys_next = parameter_sim.storage.LinearizedSystem
    op_p = linear_operator(lsys_next)
    # In-place version of:
    # ∇G += op_p*λ
    mul!(∇G, op_p, λ, 1.0, 1.0)
end

function adjoint_reassemble!(sim, state, state0, dt, forces)
    # Deal with state0 first
    reset_previous_state!(sim, state0)
    # TODO: Think this one is missing for multimodel?
    update_secondary_variables_state!(sim.storage.state0, sim.model)
    # Then the current primary variables
    reset_primary_variables!(sim.storage, sim.model, state)
    update_state_dependents!(sim.storage, sim.model, dt, forces)
    # Finally update the system
    update_linearized_system!(sim.storage, sim.model)
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

function solve_numerical_sensitivities(model, states, reports, G, target; forces = setup_forces(model), state0 = setup_state(model), parameters = setup_parameters(model))
    timesteps = report_timesteps(reports)
    N = length(states)
    @assert length(reports) == N == length(timesteps)

    base_obj = evaluate_objective(G, model, states, timesteps, forces)

    grad_num = zeros(length(parameters[target]))
    ϵ = 1e-8
    for i in eachindex(grad_num)
        param_i = deepcopy(parameters)
        param_i[target][i] += ϵ
        sim_i = Simulator(model, state0 = copy(state0), parameters = param_i)
        states_i, reports = simulate!(sim_i, timesteps, info_level = 1, forces = forces)
        v = evaluate_objective(G, model, states_i, timesteps, forces)
        grad_num[i] = (v - base_obj)/ϵ
    end
    return grad_num
end

function evaluate_objective(G, model, states, timesteps, all_forces)
    obj = 0.0
    for (step_no, state) in enumerate(states)
        forces = forces_for_timestep(nothing, all_forces, timesteps, step_no)
        dt = timesteps[step_no]
        obj += G(model, state, dt, step_no, forces)
    end
    return obj
end
