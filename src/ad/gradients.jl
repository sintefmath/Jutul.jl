export state_gradient, solve_adjoint_sensitivities, solve_adjoint_sensitivities!, setup_adjoint_storage

"""
    solve_adjoint_sensitivities(model, states, reports_or_timesteps, G; extra_timing = false, state0 = setup_state(model), forces = setup_forces(model), raw_output = false, kwarg...)

Compute sensitivities of `model` parameter with name `target` for objective function `G`.

The objective function is at the moment assumed to be a sum over all states on the form:
`obj = Σₙ G(model, state, dt_n, n, forces_for_step_n)`

Solves the adjoint equations: For model equations F the gradient with respect to parameters p is
    ∇ₚG = Σₙ (∂Fₙ / ∂p)ᵀ λₙ where n ∈ [1, N].
Given Lagrange multipliers λₙ from the adjoint equations
    (∂Fₙ / ∂xₙ)ᵀ λₙ = - (∂J / ∂xₙ)ᵀ - (∂Fₙ₊₁ / ∂xₙ)ᵀ λₙ₊₁
where the last term is omitted for step n = N and G is the objective function.
"""
function solve_adjoint_sensitivities(model, states, reports_or_timesteps, G;
        n_objective = nothing,
        extra_timing = false,
        state0 = setup_state(model),
        forces = setup_forces(model),
        raw_output = false,
        extra_output = false,
        info_level = 0,
        kwarg...
    )
    # One simulator object for the equations with respect to primary (at previous time-step)
    # One simulator object for the equations with respect to parameters
    set_global_timer!(extra_timing)
    # Allocation part
    if info_level > 1
        jutul_message("Adjoints", "Setting up storage...", color = :blue)
    end
    t_storage = @elapsed storage = setup_adjoint_storage(model; state0 = state0, n_objective = n_objective, info_level = info_level, kwarg...)
    if info_level > 1
        jutul_message("Adjoints", "Storage set up in $(get_tstr(t_storage)).", color = :blue)
    end
    parameter_model = storage.parameter.model
    n_param = number_of_degrees_of_freedom(parameter_model)
    ∇G = gradient_vec_or_mat(n_param, n_objective)
    # Timesteps
    N = length(states)
    if eltype(reports_or_timesteps)<:Real
        timesteps = reports_or_timesteps
    else
        @assert length(reports_or_timesteps) == N
        timesteps = report_timesteps(reports_or_timesteps)
    end
    @assert length(timesteps) == N "Recieved $(length(timesteps)) timesteps and $N states. These should match."
    # Solve!
    if info_level > 1
        jutul_message("Adjoints", "Solving $N adjoint steps...", color = :blue)
    end
    t_solve = @elapsed solve_adjoint_sensitivities!(∇G, storage, states, state0, timesteps, G, forces = forces, info_level = info_level)
    if info_level > 1
        jutul_message("Adjoints", "Adjoints solved in $(get_tstr(t_solve)).", color = :blue)
    end
    print_global_timer(extra_timing; text = "Adjoint solve detailed timing")
    if raw_output
        out = ∇G
    else
        if info_level > 1
            jutul_message("Adjoints", "Storing sensitivities.", color = :blue)
        end
        out = store_sensitivities(parameter_model, ∇G, storage.parameter_map)
        s0_map = storage.state0_map
        if !ismissing(s0_map)
            store_sensitivities!(out, storage.backward.model, storage.dstate0, s0_map)
        end
    end
    if extra_output
        return (out, storage)
    else
        return out
    end
end

function solve_adjoint_sensitivities(case::JutulCase, states::Vector, G; dt = case.dt, forces = case.forces, kwarg...)
    return solve_adjoint_sensitivities(
        case.model, states, dt, G;
        state0 = case.state0,
        forces = forces,
        parameters = case.parameters,
        kwarg...
    )
end

function solve_adjoint_sensitivities(case::JutulCase, some_kind_of_result, G; kwarg...)
    if hasproperty(some_kind_of_result, :result)
        simresult = some_kind_of_result.result
    else
        simresult = some_kind_of_result
    end
    simresult::SimResult
    states, dt, step_ix = expand_to_ministeps(simresult)
    forces = case.forces
    if forces isa Vector
        forces = forces[step_ix]
    end
    return solve_adjoint_sensitivities(case, states, G; dt = dt, forces = forces, kwarg...)
end

"""
    setup_adjoint_storage(model; state0 = setup_state(model), parameters = setup_parameters(model))

Set up storage for use with `solve_adjoint_sensitivities!`.
"""
function setup_adjoint_storage(model;
        state0 = setup_state(model),
        parameters = setup_parameters(model),
        n_objective = nothing,
        targets = parameter_targets(model),
        include_state0 = false,
        use_sparsity = true,
        linear_solver = select_linear_solver(model, mode = :adjoint, rtol = 1e-6),
        param_obj = true,
        info_level = 0,
        kwarg...
    )
    # Create parameter model for ∂Fₙ / ∂p
    parameter_model = adjoint_parameter_model(model, targets)
    n_prm = number_of_degrees_of_freedom(parameter_model)
    # Note that primary is here because the target parameters are now the primaries for the parameter_model
    parameter_map, = variable_mapper(parameter_model, :primary, targets = targets; kwarg...)
    if include_state0
        state0_map, = variable_mapper(model, :primary)
        n_state0 = number_of_degrees_of_freedom(model)
        state0_vec = zeros(n_state0)
    else
        state0_map = missing
        state0_vec = missing
        n_state0 = 0
    end
    # Transfer over parameters and state0 variables since many parameters are now variables
    state0_p = swap_variables(state0, parameters, parameter_model, variables = true)
    parameters_p = swap_variables(state0, parameters, parameter_model, variables = false)
    parameter_sim = Simulator(parameter_model, state0 = deepcopy(state0_p), parameters = deepcopy(parameters_p), mode = :sensitivities, extra_timing = nothing)
    if param_obj
        n_param = number_of_degrees_of_freedom(parameter_model)
        dobj_dparam = gradient_vec_or_mat(n_param, n_objective)
        param_buf = similar(dobj_dparam)
    else
        dobj_dparam = nothing
        param_buf = nothing
    end
    # Set up the generic adjoint storage
    storage = setup_adjoint_storage_base(
            model, state0, parameters,
            use_sparsity = use_sparsity,
            linear_solver = linear_solver,
            n_objective = n_objective,
            info_level = info_level,
    )
    storage[:dparam] = dobj_dparam
    storage[:param_buf] = param_buf
    storage[:parameter] = parameter_sim
    storage[:parameter_map] = parameter_map
    storage[:state0_map] = state0_map
    storage[:dstate0] = state0_vec
    storage[:n] = n_prm

    return storage
end

function setup_adjoint_storage_base(model, state0, parameters;
        use_sparsity = true,
        linear_solver = select_linear_solver(model, mode = :adjoint, rtol = 1e-8),
        n_objective = nothing,
        info_level = 0
    )
    primary_model = adjoint_model_copy(model)
    # Standard model for: ∂Fₙᵀ / ∂xₙ
    forward_sim = Simulator(primary_model, state0 = deepcopy(state0), parameters = deepcopy(parameters), mode = :forward, extra_timing = nothing)
    # Same model, but adjoint for: ∂Fₙ₊₁ᵀ / ∂xₙ
    backward_sim = Simulator(primary_model, state0 = deepcopy(state0), parameters = deepcopy(parameters), mode = :reverse, extra_timing = nothing)
    if use_sparsity isa Bool
        if use_sparsity
            # We will update these later on
            sparsity_obj = Dict{Any, Any}(
                :parameter => nothing, 
                :forward => nothing
            )
        else
            sparsity_obj = nothing
        end
    else
        # Assume it was manually set up
        sparsity_obj = use_sparsity
    end
    n_pvar = number_of_degrees_of_freedom(model)
    λ = gradient_vec_or_mat(n_pvar, n_objective)
    fsim_s = forward_sim.storage
    rhs = vector_residual(fsim_s.LinearizedSystem)
    dx = fsim_s.LinearizedSystem.dx_buffer
    n_var = length(dx)

    rhs_transfer_needed = length(rhs) != n_var
    multiple_rhs = !isnothing(n_objective)
    if multiple_rhs
        # Need bigger buffers for multiple rhs
        rhs = gradient_vec_or_mat(n_var, n_objective)
        dx = gradient_vec_or_mat(n_var, n_objective)
    elseif rhs_transfer_needed
        rhs = zeros(n_var)
    end
    storage = JutulStorage()
    storage[:forward] = forward_sim
    storage[:backward] = backward_sim
    storage[:objective_sparsity] = sparsity_obj
    storage[:lagrange] = λ
    storage[:lagrange_buffer] = similar(λ)
    storage[:dx] = dx
    storage[:rhs] = rhs
    storage[:n_forward] = n_var
    storage[:multiple_rhs] = multiple_rhs
    storage[:rhs_transfer_needed] = rhs_transfer_needed
    storage[:forward_config] = simulator_config(forward_sim, linear_solver = linear_solver, info_level = info_level)

    return storage
end

"""
    solve_adjoint_sensitivities!(∇G, storage, states, state0, timesteps, G; forces)

Non-allocating version of `solve_adjoint_sensitivities`.
"""
function solve_adjoint_sensitivities!(∇G, storage, states, state0, timesteps, G; forces, info_level = 0)
    G = adjoint_wrap_objective(G, storage.forward.model)
    packed_steps = AdjointPackedResult(states, timesteps, forces)
    packed_steps.state0 = state0
    # Do sparsity detection if not already done.
    if info_level > 1
        jutul_message("Adjoints", "Updating sparsity patterns.", color = :blue)
    end
    N = length(packed_steps)
    update_objective_sparsity!(storage, G, packed_steps, :forward)
    update_objective_sparsity!(storage, G, packed_steps, :parameter)
    # Set gradient to zero before solve starts
    @. ∇G = 0
    @tic "sensitivities" for i in N:-1:1
        if info_level > 0
            jutul_message("Step $i/$N", "Solving adjoint system.", color = :blue)
        end
        s0, s, s_next = adjoint_step_state_triplet(packed_steps, i)
        update_sensitivities!(∇G, i, G, storage, s0, s, s_next, packed_steps)
    end
    dparam = storage.dparam
    if !isnothing(dparam)
        @. ∇G += dparam
    end
    rescale_sensitivities!(∇G, storage.parameter.model, storage.parameter_map)
    @assert all(isfinite, ∇G)
    # Finally deal with initial state gradients
    if !ismissing(storage.state0_map)
        ps1 = packed_steps[1]
        forces1 = ps1.forces
        dt1 = ps1.step_info[:dt]
        adjoint_reassemble!(storage.backward, ps1.state, state0, dt1, forces1, dt1)
        update_state0_sensitivities!(storage)
    end
    return ∇G
end

function adjoint_reset_parameters!(storage, parameters)
    if haskey(storage, :parameter)
        sims = (storage.forward, storage.backward, storage.parameter)
    else
        sims = (storage.forward, storage.backward)
    end
    for sim in sims
        for k in (:state, :state0, :parameters)
            reset_variables!(sim, parameters, type = k)
        end
    end
    return storage
end

function adjoint_step_state_triplet(packed_steps::AdjointPackedResult, i::Int)
    states = packed_steps.states
    if i == 1
        s0 = packed_steps.state0
    else
        s0 = states[i-1]
    end
    s = states[i]
    if i == length(states)
        s_next = nothing
    else
        s_next = states[i+1]
    end
    return (s0, s, s_next)
end

function update_objective_sparsity!(storage, G, packed_steps::AdjointPackedResult, k = :forward)
    obj_sparsity = storage.objective_sparsity
    if isnothing(obj_sparsity) || (k == :parameter && isnothing(storage.dparam))
        return
    else
        sparsity = obj_sparsity[k]
        if isnothing(sparsity)
            sim = storage[k]
            obj_sparsity[k] = determine_objective_sparsity(sim, sim.model, G, packed_steps)
        end
    end
end

function get_objective_sparsity(storage, k)
    obj_sparsity = storage.objective_sparsity
    if isnothing(obj_sparsity)
        S = nothing
    else
        S = obj_sparsity[k]
    end
    return S
end

function determine_objective_sparsity(sim, model, G::AbstractSumObjective, packed_steps::AdjointPackedResult)
    update_secondary_variables!(sim.storage, sim.model)
    state = sim.storage.state

    function F_outer(state, i)
        ps = packed_steps[i]
        si = ps.step_info
        f = ps.forces
        return G(model, state, si[:dt], si, f)
    end
    sparsity = missing
    for i in 1:length(packed_steps)
        s_new = determine_sparsity_simple(s -> F_outer(s, i), model, state)
        sparsity = merge_sparsity!(sparsity, s_new)
    end
    return sparsity
end

function determine_objective_sparsity(sim, model, G::AbstractGlobalObjective, packed_steps::AdjointPackedResult)
    update_secondary_variables!(sim.storage, sim.model)
    state = sim.storage.state
    function F_outer(state)
        si = packed_steps.step_infos
        f = packed_steps.forces
        states = [state for _ in 1:length(packed_steps.states)]
        return G(model, state, states, si, f, packed_steps.input_data)
    end
    sparsity = determine_sparsity_simple(F_outer, model, state)
    return sparsity
end

function merge_sparsity!(::Missing, s)
    return s
end

function merge_sparsity!(sparsity::AbstractDict, s_new::AbstractDict)
    for (k, v) in s_new
        merge_sparsity!(sparsity[k], v)
    end
    return sparsity
end

function merge_sparsity!(sparsity::AbstractVector, s_new::AbstractVector)
    for k in s_new
        push!(sparsity, k)
    end
    return unique!(sparsity)
end

function state_gradient(model, state, F, extra_arg...; kwarg...)
    n_total = number_of_degrees_of_freedom(model)
    ∂F∂x = zeros(n_total)
    return state_gradient!(∂F∂x, model, state, F, extra_arg...; kwarg...)
end

function state_gradient!(∂F∂x, model, state, F, extra_arg...; parameters = setup_parameters(model))
    # Either with respect to all primary variables, or all parameters.
    state = setup_state(model, state)
    state = merge_state_with_parameters(model, state, parameters)
    state = convert_state_ad(model, state)
    state = convert_to_immutable_storage(state)
    update_secondary_variables_state!(state, model)
    state_gradient_outer!(∂F∂x, F, model, state, extra_arg)
    return ∂F∂x
end

function merge_state_with_parameters(model, state, parameters)
    for (k, v) in pairs(parameters)
        state[k] = v
    end
    return state
end

function state_gradient_outer!(∂F∂x, obj_eval, model, state; sparsity = nothing)
    if !isnothing(sparsity)
        @. ∂F∂x = 0
    end
    state_gradient_inner!(∂F∂x, obj_eval, model, state, nothing; sparsity = sparsity)
end

function state_gradient_inner!(∂F∂x, F, model, state, tag, eval_model = model; sparsity = nothing, symbol = nothing)
    layout = matrix_layout(model.context)
    get_partial(x::AbstractFloat, i) = 0.0
    get_partial(x::ForwardDiff.Dual, i) = x.partials[i]

    function store_partials!(∂F∂x::AbstractVector, v, i, ne, np, offset)
        for p_i in 1:np
            ix = alignment_linear_index(i, p_i, ne, np, layout) + offset
            ∂F∂x[ix] = get_partial(v, p_i)
        end
    end

    function store_partials!(∂F∂x::AbstractMatrix, v, i, ne, np, offset)
        for j in 1:size(∂F∂x, 2)
            for p_i in 1:np
                ix = alignment_linear_index(i, p_i, ne, np, layout) + offset
                ∂F∂x[ix, j] = get_partial(v[j], p_i)
            end
        end
    end

    function diff_entity!(∂F∂x, state, i, S, ne, np, offset, symbol)
        state_i = local_ad(state, i, S, symbol)
        v = F(eval_model, state_i)
        store_partials!(∂F∂x, v, i, ne, np, offset)
    end

    offset = 0
    for e in get_primary_variable_ordered_entities(model)
        np = number_of_partials_per_entity(model, e)
        ne = count_active_entities(model.domain, e)
        if isnothing(sparsity)
            it_rng = 1:ne
        else
            it_rng = sparsity[e]
        end
        if length(it_rng) > 0
            ltag = get_entity_tag(tag, e)
            S = typeof(get_ad_entity_scalar(1.0, np, tag = ltag))
            for i in it_rng
                diff_entity!(∂F∂x, state, i, S, ne, np, offset, symbol)
            end
        end
        offset += ne*np
    end
end

function update_sensitivities!(∇G, i, G, adjoint_storage, state0, state, state_next, packed_steps::AdjointPackedResult)
    N = length(packed_steps)
    # Figure out when we are in discrete index and time
    packed_step = packed_steps[i]
    step_info = packed_step.step_info
    dt = step_info[:dt]
    current_time = step_info[:time]
    report_step = step_info[:step]
    forces = packed_step.forces
    for skey in [:backward, :forward, :parameter]
        s = adjoint_storage[skey]
        recorder_reset!(progress_recorder(s), step = report_step, time = current_time)
    end
    λ = next_lagrange_multiplier!(adjoint_storage, i, G, state0, state, state_next, packed_steps)
    # λ, t, dt, forces
    @assert all(isfinite, λ) "Non-finite lagrange multiplier found in step $i. Linear solver failure?"
    # ∇ₚG = Σₙ (∂Fₙ / ∂p)ᵀ λₙ
    # Increment gradient
    parameter_sim = adjoint_storage.parameter
    @tic "jacobian (for parameters)" adjoint_reassemble!(parameter_sim, state, state0, dt, forces, current_time)
    lsys_param = parameter_sim.storage.LinearizedSystem
    op_p = linear_operator(lsys_param)
    sens_add_mult!(∇G, op_p, λ)
    dparam = adjoint_storage.dparam
    @tic "objective parameter gradient" if !isnothing(dparam)
        if i == N
            @. dparam = 0
        end
        obj_eval = objective_evaluator_from_model_and_state(G, parameter_sim.model, packed_steps, i)
        pbuf = adjoint_storage.param_buf
        S_p = get_objective_sparsity(adjoint_storage, :parameter)
        state_gradient_outer!(pbuf, obj_eval, parameter_sim.model, parameter_sim.storage.state, sparsity = S_p)
        @. dparam += pbuf
    end
end

function next_lagrange_multiplier!(adjoint_storage, i, G, state0, state, state_next, packed_steps::AdjointPackedResult)
    # Unpack simulators
    backward_sim = adjoint_storage.backward
    forward_sim = adjoint_storage.forward
    λ = adjoint_storage.lagrange
    λ_b = adjoint_storage.lagrange_buffer
    config = adjoint_storage.forward_config

    packed_step = packed_steps[i]
    step_info = packed_step.step_info
    dt = step_info[:dt]
    forces = packed_step.forces
    N = length(packed_steps)

    # Assemble Jacobian w.r.t. current step
    @tic "jacobian (standard)" adjoint_reassemble!(forward_sim, state, state0, dt, forces, step_info[:time])
    il = config[:info_level]
    converged, e, errors = check_convergence(
        forward_sim.storage,
        forward_sim.model,
        config,
        iteration = 1,
        dt = dt,
        extra_out = true
    )
    if !converged && il > 0
        jutul_message("Warning", "Simulation was not converged to default tolerances for step $i in adjoint solve", color = :yellow)
        if il > 1.5
            get_convergence_table(errors, il, 1, config)
        end
    end

    # Note the sign: There is an implicit negative sign in the linear solver when solving for the Newton increment. Therefore, the terms of the
    # right hand side are added with a positive sign instead of negative.
    rhs = adjoint_storage.rhs
    # Fill rhs with (∂J / ∂x)ᵀₙ (which will be treated with a negative sign when the result is written by the linear solver)
    S_p = get_objective_sparsity(adjoint_storage, :forward)
    obj_eval = objective_evaluator_from_model_and_state(G, forward_sim.model, packed_steps, i)
    @tic "objective primary gradient" state_gradient_outer!(rhs, obj_eval, forward_sim.model, forward_sim.storage.state, sparsity = S_p)
    if isnothing(state_next)
        @assert i == N
        @. λ = 0
    else
        next_packed_step = packed_steps[i+1]
        next_step_info = next_packed_step.step_info
        forces_next = next_packed_step.forces
        @tic "jacobian (with state0)" adjoint_reassemble!(backward_sim, state_next, state, next_step_info[:dt], forces_next, next_step_info[:time])
        lsys_next = backward_sim.storage.LinearizedSystem
        op = linear_operator(lsys_next, skip_red = true)
        # In-place version of
        # rhs += op*λ
        # - (∂Fₙ₊₁ / ∂xₙ)ᵀ λₙ₊₁
        adjoint_transfer_canonical_order!(λ_b, λ, forward_sim.model, to_canonical = false)
        sens_add_mult!(rhs, op, λ_b)
    end
    # We have the right hand side, assemble the Jacobian and solve for the Lagrange multiplier
    lsolve = adjoint_storage.forward_config[:linear_solver]
    dx = adjoint_storage.dx
    if adjoint_storage.multiple_rhs
        lsolve_arg = (dx = dx, r = rhs)
    else
        lsolve_arg = NamedTuple()
    end
    lsys = forward_sim.storage.LinearizedSystem
    if adjoint_storage.rhs_transfer_needed
        lsys.r_buffer .= rhs
    end
    @tic "linear solve" lstats = linear_solve!(lsys, lsolve, forward_sim.model, forward_sim.storage; lsolve_arg...)
    adjoint_transfer_canonical_order!(λ, dx, forward_sim.model, to_canonical = true)
    return λ
end

function sens_add_mult!(x::AbstractVector, op::LinearOperator, y::AbstractVector)
    mul!(x, op, y, 1.0, 1.0)
end

function sens_add_mult!(x::AbstractMatrix, op::LinearOperator, y::AbstractMatrix)
    for i in axes(x, 2)
        x_i = vec(view(x, :, i))
        y_i = vec(view(y, :, i))
        sens_add_mult!(x_i, op, y_i)
    end
end

function adjoint_reassemble!(sim, state, state0, dt, forces, time)
    s = sim.storage
    model = sim.model
    # Deal with state0 first
    reset_previous_state!(sim, state0)
    update_secondary_variables!(s, model, true)
    # Make sure that state is that of the previous state TODO: This does an
    # extra update of properties that could maybe be avoided, but is needed to
    # make everything consistent with how the forward simulator works before
    # update_before_step! is called.
    reset_state_to_previous_state!(sim)
    # Apply logic as if timestep is starting
    update_before_step!(s, model, dt, forces, time = time, recorder = progress_recorder(sim))
    # Then the current primary variables
    reset_variables!(s, model, state)
    update_state_dependents!(s, model, dt, forces, time = time, update_secondary = true)
    # Finally update the system
    update_linearized_system!(s, model)
end

function swap_primary_with_parameters!(pmodel, model, targets = parameter_targets(model))
    for (k, v) in pairs(model.primary_variables)
        if !(k in targets)
            set_parameters!(pmodel; k => v)
        end
    end
    # Original model holds the parameters, use those
    for (k, v) in pairs(model.parameters)
        if k in targets
            set_primary_variables!(pmodel; k => v)
        end
    end
    out = vcat(keys(model.secondary_variables)..., keys(model.primary_variables)...)
    for k in out
        push!(pmodel.output_variables, k)
    end
    unique!(pmodel.output_variables)
    return pmodel
end

function parameter_targets(model::SimulationModel)
    prm = get_parameters(model)
    targets = Symbol[]
    for (k, v) in prm
        if parameter_is_differentiable(v, model)
            push!(targets, k)
        end
    end
    return targets
end

function adjoint_parameter_model(model, arg...; context = DefaultContext())
    # By default the adjoint model uses the default context since no linear solver
    # is needed.
    pmodel = adjoint_model_copy(model; context = context)
    # Swap parameters and primary variables
    swap_primary_with_parameters!(pmodel, model, arg...)
    ensure_model_consistency!(pmodel)
    return sort_variables!(pmodel, :all)
end

function adjoint_model_copy(model::SimulationModel{O, S, F, C}; context = model.context) where {O, S, C, F}
    pvar = copy(model.primary_variables)
    svar = copy(model.secondary_variables)
    outputs = vcat(keys(pvar)..., keys(svar)...)
    prm = copy(model.parameters)
    extra = deepcopy(model.extra)
    eqs = model.equations
    # Transpose the system
    new_context = adjoint(context)
    N = typeof(new_context)
    return SimulationModel{O, S, F, N}(
        model.domain,
        model.system,
        new_context,
        model.formulation,
        model.data_domain,
        pvar,
        svar,
        prm,
        eqs,
        outputs,
        extra
    )
end

"""
    solve_numerical_sensitivities(model, states, reports, G, target;
                                                forces = setup_forces(model),
                                                state0 = setup_state(model),
                                                parameters = setup_parameters(model),
                                                epsilon = 1e-8)

Compute sensitivities of `model` parameter with name `target` for objective function `G`.

This method uses numerical perturbation and is primarily intended for testing of `solve_adjoint_sensitivities`.
"""
function solve_numerical_sensitivities(model, states, reports, G, target;
                                                forces = setup_forces(model),
                                                state0 = setup_state(model),
                                                parameters = setup_parameters(model),
                                                epsilon = 1e-8)
    timesteps = report_timesteps(reports)
    G = adjoint_wrap_objective(G, model)
    N = length(states)
    @assert length(reports) == N == length(timesteps)
    # Base objective
    base_obj = evaluate_objective(G, model, states, timesteps, forces)
    # Define perturbation
    param_var, param_num = get_parameter_pair(model, parameters, target)
    sz = size(param_num)
    grad_num = zeros(sz)
    grad_is_vector = grad_num isa AbstractVector
    if grad_is_vector
        grad_num = grad_num'
        sz = size(grad_num)
    end
    scale = variable_scale(param_var)
    if isnothing(scale)
        ϵ = epsilon
    else
        ϵ = scale*epsilon
    end
    n, m = sz
    for i in 1:n
        for j in 1:m
            param_i = deepcopy(parameters)
            perturb_parameter!(model, param_i, target, i, j, sz, ϵ)
            sim_i = Simulator(model, state0 = copy(state0), parameters = param_i)
            states_i, reports = simulate!(sim_i, timesteps, info_level = -1, forces = forces)
            v = evaluate_objective(G, model, states_i, timesteps, forces)
            grad_num[i, j] = (v - base_obj)/ϵ
        end
    end
    if grad_is_vector
        grad_num = grad_num'
    end
    return grad_num
end

function solve_numerical_sensitivities(model, states, reports, G; kwarg...)
    out = Dict()
    for k in keys(model.parameters)
        out[k] = solve_numerical_sensitivities(model, states, reports, G, k; kwarg...)
    end
end

function get_parameter_pair(model, parameters, target)
    return (model.parameters[target], parameters[target])
end

function perturb_parameter!(model, param_i, target, i, j, sz, ϵ)
    param_i[target][j + i*(sz[1]-1)] += ϵ
end

function evaluate_objective(G, model, states, timesteps, all_forces;
        step_index = eachindex(states),
        kwarg...
    )
    packed_steps = AdjointPackedResult(states, timesteps, all_forces, step_index)
    obj = adjoint_wrap_objective(G, model)
    return evaluate_objective(obj, model, packed_steps; kwarg...)
end

function evaluate_objective(G::AbstractSumObjective, model, packed_steps::AdjointPackedResult;
        kwarg...
    )
    obj = 0.0
    for i in 1:length(packed_steps)
        packed = packed_steps[i]
        si = packed.step_info
        obj += G(
            model,
            packed.state,
            si[:dt],
            si,
            packed.forces
        )
    end
    return obj
end

function evaluate_objective(G::AbstractGlobalObjective, model, packed_steps::AdjointPackedResult; input_data = missing)
    return G(model, packed_steps.state0, packed_steps.states, packed_steps.step_infos, packed_steps.forces, input_data)
end

function store_sensitivities(model, result, prm_map)
    out = Dict{Symbol, Any}()
    store_sensitivities!(out, model, result, prm_map)
    return out
end

function store_sensitivities!(out, model, result, prm_map; kwarg...)
    variables = model.primary_variables
    layout = matrix_layout(model.context)
    return store_sensitivities!(out, model, variables, result, prm_map, layout; kwarg...)
end

function store_sensitivities!(out, model, variables, result, prm_map, ::EquationMajorLayout)
    scalar_valued_objective = result isa AbstractVector
    if scalar_valued_objective
        N = 1
    else
        N = size(result, 2)
    end
    for (k, var) in pairs(variables)
        if !haskey(prm_map, k)
            continue
        end
        (; n_x, offset_x) = prm_map[k]
        m = degrees_of_freedom_per_entity(model, var)
        if n_x > 0
            rng = (offset_x+1):(offset_x+n_x)
            if scalar_valued_objective
                result_k = view(result, rng)
                v = extract_sensitivity_subset(result_k, var, n_x, m, offset_x)
            else
                # Grab each of the sensitivities and put them in an array for simplicity
                # TODO: Update this part
                v = map(
                    i -> extract_sensitivity_subset(view(result, rng, i), var, n_x, m, n_x*(i-1)),
                    1:N
                )
            end
            out[k] = v
        else
            out[k] = similar(result, 0)
        end
    end
    return out
end

function store_sensitivities!(out, model, variables, result, prm_map, ::Union{EquationMajorLayout, BlockMajorLayout})
    scalar_valued_objective = result isa AbstractVector
    @assert scalar_valued_objective "Only supported for scalar objective"

    us = get_primary_variable_ordered_entities(model)
    @assert length(us) == 1 "This function is not implemented for more than one entity type for primary variables"
    u = only(us)
    bz = degrees_of_freedom_per_entity(model, u)
    ne = count_active_entities(model.domain, u)

    offset = 1
    for (k, var) in pairs(variables)
        m = degrees_of_freedom_per_entity(model, var)
        var::ScalarVariable
        pos = offset:bz:(bz*(ne-1)+offset)
        @assert length(pos) == ne "$(length(pos))"
        out[k] = result[pos]
        offset += 1
    end
    return out
end

function extract_sensitivity_subset(r, var, n, m, offset)
    if var isa ScalarVariable
        v = r
    else
        v = reshape(r, n ÷ m, m)'
    end
    v = collect(v)
    return v
end

gradient_vec_or_mat(n, ::Nothing) = zeros(n)
gradient_vec_or_mat(n, m) = zeros(n, m)

function variable_mapper(model::SimulationModel, type = :primary;
        targets = nothing,
        config = nothing,
        offset_x = 0,
        offset_full = offset_x
    )
    vars = get_variables_by_type(model, type)
    out = OrderedDict{Symbol, Any}()
    for (t, var) in vars
        if !isnothing(targets) && !(t in targets)
            continue
        end
        var = vars[t]
        if var isa Pair
            var = last(var)
        end
        n = number_of_values(model, var)
        m = values_per_entity(model, var)
        n_x = n
        if !isnothing(config)
            lumping = config[t][:lumping]
            if !isnothing(lumping)
                N = maximum(lumping)
                sort(unique(lumping)) == 1:N || error("Lumping for $t must include all values from 1 to $N")
                length(lumping) == n÷m || error("Lumping for $t must have one entry for each value if present")
                n_x = N*m
            end
        end
        out[t] = (
            n_full = n,
            n_x = n_x,
            n_row = m,
            offset_full = offset_full,
            offset_x = offset_x,
            scale = variable_scale(var)
        )
        offset_full += n
        offset_x += n_x
    end
    return (out, offset_full, offset_x)
end

function rescale_sensitivities!(dG, model, parameter_map; renum = nothing)
    for (k, v) in parameter_map
        (; n_full, offset_full, scale) = v
        if !isnothing(scale)
            interval = (offset_full+1):(offset_full+n_full)
            if !isnothing(renum)
                interval = renum[interval]
            end
            if dG isa AbstractVector
                dG_k = view(dG, interval)
            else
                dG_k = view(dG, interval, :)
            end
            @. dG_k /= scale
        end
    end
end

function swap_variables(state, parameters, model; variables = true)
    p_keys = keys(get_parameters(model))
    v_keys = keys(get_variables(model))

    out = Dict{Symbol, Any}()
    # If variables: state should contain all fields, except those that are now in parameters
    if variables
        internal_swapper!(out, state, parameters, v_keys, p_keys)
    else
        internal_swapper!(out, parameters, state, p_keys, v_keys)
    end
    return out
end

function internal_swapper!(out, A, B, keep, skip)
    for (k, v) in A
        if !(k in skip)
            out[k] = v
        end
    end
    for (k, v) in B
        if k in keep
            out[k] = v
        end
    end
end

function adjoint_transfer_canonical_order!(λ, dx, model::MultiModel; to_canonical = true)
    offset = 0
    for (k, m) in pairs(model.models)
        ndof = number_of_degrees_of_freedom(m)
        ix = (offset+1):(offset+ndof)
        λ_i = view(λ, ix)
        dx_i = view(dx, ix)
        adjoint_transfer_canonical_order_inner!(λ_i, dx_i, m, matrix_layout(m.context), to_canonical)
        offset += ndof
    end
end

function adjoint_transfer_canonical_order!(λ, dx, model; to_canonical = true)
    adjoint_transfer_canonical_order_inner!(λ, dx, model, matrix_layout(model.context), to_canonical)
end

function adjoint_transfer_canonical_order_inner!(λ, dx, model, ::EquationMajorLayout, to_canonical)
    @. λ = dx
end

function adjoint_transfer_canonical_order_inner!(λ, dx, model, ::BlockMajorLayout, to_canonical)
    bz = 0
    for e in get_primary_variable_ordered_entities(model)
        if bz > 0
            error("Assumed that block major has a single entity group")
        end
        bz = degrees_of_freedom_per_entity(model, e)::Int
    end
    n = length(dx) ÷ bz
    n::Int
    if to_canonical
        for b in 1:bz
            for i in 1:n
                λ[(b-1)*n + i] = dx[(i-1)*bz + b]
            end
        end
    else
        for b in 1:bz
            for i in 1:n
                λ[(i-1)*bz + b] = dx[(b-1)*n + i]
            end
        end
    end
end

function update_state0_sensitivities!(storage)
    state0_map = storage.state0_map
    if !ismissing(state0_map)
        sim = storage.backward
        model = sim.model
        # Assume that this gets called at the end when everything has been set
        # up in terms of the simulators
        λ = storage.lagrange # has canonical order
        order = collect(eachindex(λ))
        renum = similar(order) # use for reordering λ and scaling
        adjoint_transfer_canonical_order!(renum, order, model)
        λ_renum = similar(λ)
        @. λ_renum[renum] = λ
        lsys_b = sim.storage.LinearizedSystem
        op_b = linear_operator(lsys_b, skip_red = true)
        ∇x = storage.dstate0
        @. ∇x = 0.0
        sens_add_mult!(∇x, op_b, λ_renum)
        rescale_sensitivities!(∇x, model, storage.state0_map, renum = renum)
    end
end

"""
    optimization_step_info(step::Int, time::Real, dt; kwarg...)

Optimization step information is normally not set up manually, but the output
from this function will be passed to objective functions as `step_info`. This
function is a `Dict` with the following fields:

# Fields
- `:time` - The time at the end of the step. To get time at the start of the
  step, use `step_info[:time] - step_info[:dt]`
- `:dt` - The time step size for this step.
- `:step` - The step number, starting at 1. Not that this is the report step,
  and multiple `step_info` entries could have the same step if substeps are
  used.
- `:Nstep` - The total number of steps in the simulation. 
- `:substep` - The substep number, starting at 1. This is used to indicate
  that multiple steps are taken within a single step.
- `:substep_global` - The global substep number, starting at 1 and will not
  reset between global steps.
- `:Nsubstep_global` - The total number of substeps in the simulation.
- `:total_time` - The total time of the simulation (i.e. time at the final
  step and substep)
"""
function optimization_step_info(step::Int, time::Real, dt::Real;
        Nstep = missing,
        total_time = missing,
        substep_global = step,
        Nsubstep_global = Nstep,
        substep = 1,
        kwarg...
    )
    out = OrderedDict{Symbol, Any}(
        :time => time,
        :dt => dt,
        :step => step,
        :Nstep => Nstep,
        :substep => substep,
        :substep_global => substep_global,
        :Nsubstep_global => Nsubstep_global,
        :total_time => total_time,
    )
    for (k, v) in kwarg
        out[k] = v
    end
    return out
end

function optimization_step_info(step::Int, dts::Vector; kwarg...)
    t = 0.0
    for i in 1:(step)
        t += dts[i]
    end
    return optimization_step_info(step, t, dts[step]; Nstep = length(dts), total_time = sum(dts), kwarg...)
end
