    function solve_adjoint_generic(X, F, states, reports_or_timesteps, G;
            n_objective = nothing,
            extra_timing = false,
            extra_output = false,
            forces = missing,
            info_level = 0,
            state0 = missing,
            kwarg...
        )
        # F(X, step_info) -> model*
        # F(X, step_info) -> model, parameters*
        # F(X, step_info) -> model, parameters, forces*
        # F(X, step_info) -> case (current step)
        # F(X, step_info) -> case (all steps)
        # *state0 needs to be provided
        Jutul.set_global_timer!(extra_timing)
        # Allocation part
        if info_level > 1
            jutul_message("Adjoints", "Setting up storage...", color = :blue)
        end
        # model, state0, parameters = setup_case(X, F, step_info, forces)
        # t_storage = @elapsed storage = setup_adjoint_storage(model; state0 = state0, n_objective = n_objective, info_level = info_level, kwarg...)
        storage = setup_adjoint_storage_base(
                model, state0, parameters,
                use_sparsity = true,
                linear_solver = Jutul.select_linear_solver(model, mode = :adjoint, rtol = 1e-6),
                n_objective = n_objective,
                info_level = info_level,
        )
        if info_level > 1
            jutul_message("Adjoints", "Storage set up in $(get_tstr(t_storage)).", color = :blue)
        end
        ∇G = gradient_vec_or_mat(n_param)
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
        t_solve = @elapsed solve_adjoint_generic!(∇G, storage, states, state0, timesteps, G, forces = forces, info_level = info_level)
        if info_level > 1
            jutul_message("Adjoints", "Adjoints solved in $(get_tstr(t_solve)).", color = :blue)
        end
        print_global_timer(extra_timing; text = "Adjoint solve detailed timing")
        if extra_output
            return (∇G, storage)
        else
            return ∇G
        end
    end

    function solve_adjoint_generic!(∇G, storage, states, state0, timesteps, G; forces = setup_forces(model), info_level = 0)
    N = length(timesteps)
    @assert N == length(states)
    if forces isa Vector
        @assert length(forces) == N
    end
    # Do sparsity detection if not already done.
    if info_level > 1
        jutul_message("Adjoints", "Updating sparsity patterns.", color = :blue)
    end

    update_objective_sparsity!(storage, G, states, timesteps, forces, :forward)
    update_objective_sparsity!(storage, G, states, timesteps, forces, :parameter)
    # Set gradient to zero before solve starts
    @. ∇G = 0
    @tic "sensitivities" for i in N:-1:1
        if info_level > 0
            jutul_message("Step $i/$N", "Solving adjoint system.", color = :blue)
        end
        s, s0, s_next = state_pair_adjoint_solve(state0, states, i, N)
        update_sensitivities!(∇G, i, G, storage, s0, s, s_next, timesteps, forces)
    end
    dparam = storage.dparam
    if !isnothing(dparam)
        @. ∇G += dparam
    end
    rescale_sensitivities!(∇G, storage.parameter.model, storage.parameter_map)
    @assert all(isfinite, ∇G)
    # Finally deal with initial state gradients
    update_state0_sensitivities!(storage)
    return ∇G
end

function setup_case(x, step_info, forces)

    # F(X, step_info) -> model
    # F(X, step_info) -> case (current step)
    # F(X, step_info) -> case (all steps)
end