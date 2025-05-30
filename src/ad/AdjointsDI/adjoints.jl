    function solve_adjoint_generic(X, F, states, reports_or_timesteps, G;
            # n_objective = nothing,
            extra_timing = false,
            extra_output = false,
            info_level = 0,
            kwarg...
        )
        Jutul.set_global_timer!(extra_timing)
        N = length(states)
        n_param = length(X)
        # Timesteps
        if eltype(reports_or_timesteps)<:Real
            timesteps = reports_or_timesteps
        else
            @assert length(reports_or_timesteps) == N
            timesteps = report_timesteps(reports_or_timesteps)
        end
        # Allocation part
        if info_level > 1
            jutul_message("Adjoints", "Setting up storage...", color = :blue)
        end
        t_storage = @elapsed storage = setup_adjoint_storage_generic(X, F, states, timesteps, G;
            info_level = info_level,
            kwarg...
        )

        if info_level > 1
            jutul_message("Adjoints", "Storage set up in $(get_tstr(t_storage)).", color = :blue)
        end
        ∇G = zeros(n_param)

        @assert length(timesteps) == N "Recieved $(length(timesteps)) timesteps and $N states. These should match."
        # Solve!
        if info_level > 1
            jutul_message("Adjoints", "Solving $N adjoint steps...", color = :blue)
        end
        t_solve = @elapsed solve_adjoint_generic!(∇G, X, storage, states, storage[:state0], timesteps, G, storage[:forces], info_level = info_level)
        if info_level > 1
            jutul_message("Adjoints", "Adjoints solved in $(get_tstr(t_solve)).", color = :blue)
        end
        Jutul.print_global_timer(extra_timing; text = "Adjoint solve detailed timing")
        if extra_output
            return (∇G, storage)
        else
            return ∇G
        end
    end

function solve_adjoint_generic!(∇G, X, storage, states, state0, timesteps, G, forces; info_level = 0)
    F_eval = storage[:function_di]
    N = length(timesteps)
    @assert N == length(states)
    if forces isa Vector
        @assert length(forces) == N
    end
    # Do sparsity detection if not already done.
    if info_level > 1
        jutul_message("Adjoints", "Updating sparsity patterns.", color = :blue)
    end

    Jutul.update_objective_sparsity!(storage, G, states, timesteps, forces, :forward)
    # Set gradient to zero before solve starts
    @. ∇G = 0
    @tic "sensitivities" for i in N:-1:1
        if info_level > 0
            jutul_message("Step $i/$N", "Solving adjoint system.", color = :blue)
        end
        s, s0, s_next = Jutul.state_pair_adjoint_solve(state0, states, i, N)
        update_sensitivities_generic!(∇G, X, F_eval, i, G, storage, s0, s, s_next, timesteps, forces)
    end
    dparam = storage.dparam
    if !isnothing(dparam)
        @. ∇G += dparam
    end
    all(isfinite, ∇G) || error("Adjoint solve resulted in non-finite gradient values.")
    return ∇G
end

function setup_adjoint_storage_generic(X, F, states, timesteps, G;
        forces = missing,
        state0 = missing,
        backend = missing,
        do_prep = true,
        di_sparse = true,
        info_level = 0,
        single_step_sparsity = true,
        use_sparsity = true
    )
    N = length(timesteps)
    eltype(timesteps)<:Real
    @assert length(states) == N "Received $(length(states)) states and $N timesteps. These should match."
    case_for_step(x_i, step_info) = setup_case(x_i, F, step_info, state0, forces, N)
    total_time = sum(timesteps)
    case0 = case_for_step(X, Jutul.optimization_step_info(1, 0.0, timesteps[1], Nstep = N, total_time = total_time))
    if ismissing(forces)
        forces = case0.forces
    end
    if ismissing(state0)
        state0 = case0.state0
    end
    storage = Jutul.setup_adjoint_storage_base(
            case0.model, state0, case0.parameters,
            use_sparsity = use_sparsity,
            linear_solver = Jutul.select_linear_solver(case0.model, mode = :adjoint, rtol = 1e-6),
            n_objective = nothing,
            info_level = info_level,
    )
    storage[:dparam] = zeros(length(X))
    storage[:forces] = forces
    storage[:state0] = state0
    setup_jacobian_evaluation!(storage, X, F, G, states, case0, forces, timesteps, backend, do_prep, single_step_sparsity, di_sparse)
    return storage
end

function update_sensitivities_generic!(∇G, X, F_eval, i, G, adjoint_storage, state0, state, state_next, timesteps, all_forces)
    solved_timesteps = @view timesteps[1:(i-1)]
    current_time = sum(solved_timesteps)
    for skey in [:backward, :forward]
        s = adjoint_storage[skey]
        Jutul.reset!(Jutul.progress_recorder(s), step = i, time = current_time)
    end
    λ, t, dt, forces = Jutul.next_lagrange_multiplier!(adjoint_storage, i, G, state, state0, state_next, timesteps, all_forces)
    @assert all(isfinite, λ) "Non-finite lagrange multiplier found in step $i. Linear solver failure?"

    total_time = sum(timesteps)
    step_info = Jutul.optimization_step_info(i, current_time, dt, total_time = total_time, Nstep = length(timesteps))
    F(x) = F_eval(x, state, state0, step_info, dt)
    prep = adjoint_storage[:prep_di]
    backend = adjoint_storage[:backend_di]
    if isnothing(prep)
        jac = jacobian(F, backend, X)
    else
        jac = jacobian(F, prep, backend, X)
    end
    # jac = jacobian(F, AutoForwardDiff(), X)
    # Add zero entry (corresponding to objective values) to avoid resizing matrix.
    N = length(λ)
    push!(λ, 0.0)
    # tmp = jac'*λ
    mul!(∇G, jac', λ, 1.0, 1.0)
    resize!(λ, N)
    # Gradient of partial objective to parameters
    dparam = adjoint_storage.dparam
    if i == length(timesteps)
        @. dparam = 0
    end
    dobj = jac[end, :]
    @. dparam += dobj
    return ∇G
end

function setup_case(x, F, step_info, state0, forces, N)
    # F(X, step_info) -> model*
    # F(X, step_info) -> model, parameters*
    # F(X, step_info) -> model, parameters, forces*
    # F(X, step_info) -> model, parameters, forces*, state0*
    # F(X, step_info) -> case (current step)
    # F(X, step_info) -> case (all steps)
    # *state0 needs to be provided
    c = unpack_setup(step_info, N, F(x, step_info))
    if c isa JutulCase
        case = c
    else
        model, p, f, s0 = c
        if ismissing(state0)
            state0 = s0
        end
        if ismissing(f)
            f = forces
        end
        if ismissing(p)
            p = Jutul.setup_parameters(model)
        end
        dt = [step_info[:dt]]
        case = JutulCase(model, dt, f, parameters = p, state0 = state0)
    end
    return case
end

function unpack_setup(step_info, N, case::JutulCase)
    # Either case for all steps or case for current step
    Ns = length(case.dt)
    if Ns > 1
        if case.forces isa Vector
            Ns == N || error("case.forces was a vector and expected $N steps, got $Ns.")
        end
        case = case[step_info[:step]]
    end
    return case
end

function unpack_setup(step_info, N, out::Tuple)
    unpack_setup(step_info, N, out...)
end

function unpack_setup(step_info, N, model::Jutul.JutulModel)
    return (model, missing, missing, missing)
end

function unpack_setup(step_info, N, model::Jutul.JutulModel, parameters)
    return (model, parameters, missing, missing)
end

function unpack_setup(step_info, N, model::Jutul.JutulModel, parameters, forces)
    return (model, parameters, forces, missing)
end

function unpack_setup(step_info, N, model::Jutul.JutulModel, parameters, forces, state0)
    return (model, parameters, forces, state0)
end

function setup_jacobian_evaluation!(storage, X, F, G, states, case0, forces, timesteps, backend, do_prep, single_step_sparsity, di_sparse)
    N = length(timesteps)
    if ismissing(backend)
        if di_sparse
            backend = AutoSparse(
                AutoForwardDiff();
                sparsity_detector = TracerLocalSparsityDetector(),
                coloring_algorithm = GreedyColoringAlgorithm(),
            )
        else
            backend = AutoForwardDiff()
        end
    end
    cache = Dict()
    # TODO: Pass cache to evaluate_residual_and_jacobian_for_state_pair
    evaluate_for_states(x, state, state0, step_info, dt) = evaluate_residual_and_jacobian_for_state_pair(x, state, state0, step_info, dt, F, G, forces, N)
    function evaluate0(x)
        t = 0.0
        dt_i = case0.dt[1]
        total_time = sum(timesteps)
        N = length(timesteps)
        info = Jutul.optimization_step_info(1, t, dt_i, total_time = total_time, Nstep = N)
        v = evaluate_for_states(x, states[1], case0.state0, info, dt_i)
        t += dt_i
        if !single_step_sparsity
            @. v = v^2
            for i in 2:N
                dt_i = timesteps[i]
                step_info = Jutul.optimization_step_info(i, t, dt_i, total_time = total_time, Nstep = N)
                tmp = evaluate_for_states(x, states[i], states[i-1], step_info, dt_i)
                @. v += tmp.^2
                t += dt_i
            end
        end
        return v
    end
    storage[:function_di] = evaluate_for_states
    # Note: strict = false is needed because we create another function on the fly
    # that essentially calls the same function.
    if do_prep
        storage[:prep_di] = prepare_jacobian(evaluate0, backend, X, strict=Val(false))
    else
        storage[:prep_di] = nothing
    end
    storage[:backend_di] = backend
    return storage
end

function evaluate_residual_and_jacobian_for_state_pair(x, state, state0, step_info, dt, F, G, forces, N = 1, cache = missing)
    case = setup_case(x, F, step_info, state0, forces, N)
    if step_info[:step] == 1
        state0 = case.state0
    end
    sim = HelperSimulator(case, eltype(x), cache = cache, n_extra = 1)
    model_residual(state, state0, sim,
        forces = case.forces,
        time = step_info[:time],
        dt = dt
    )
    r = sim.storage.r_extended
    r[end] = G(case.model, state, dt, step_info, case.forces)
    return copy(r)
end
