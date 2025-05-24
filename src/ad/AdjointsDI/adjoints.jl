    function solve_adjoint_generic(X, F, states, reports_or_timesteps, G;
            # n_objective = nothing,
            extra_timing = false,
            extra_output = false,
            forces = missing,
            state0 = missing,
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
        case_for_step(x_i, step_info) = setup_case(x_i, F, step_info, state0, forces, N)
        case0 = case_for_step(X, Jutul.optimization_step_info(1, 0.0, timesteps[1]))
        if ismissing(forces)
            forces = Jutul.setup_forces(case0.model)
        end
        # t_storage = @elapsed storage = setup_adjoint_storage(model; state0 = state0, n_objective = n_objective, info_level = info_level, kwarg...)
        storage = Jutul.setup_adjoint_storage_base(
                case0.model, case0.state0, case0.parameters,
                use_sparsity = true,
                linear_solver = Jutul.select_linear_solver(case0.model, mode = :adjoint, rtol = 1e-6),
                n_objective = nothing,
                info_level = info_level,
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

    Jutul.update_objective_sparsity!(storage, G, states, timesteps, forces, :forward)
    # update_objective_sparsity!(storage, G, states, timesteps, forces, :parameter)
    # Set gradient to zero before solve starts
    @. ∇G = 0
    @tic "sensitivities" for i in N:-1:1
        if info_level > 0
            jutul_message("Step $i/$N", "Solving adjoint system.", color = :blue)
        end
        s, s0, s_next = Jutul.state_pair_adjoint_solve(state0, states, i, N)
        update_sensitivities_generic!(∇G, i, G, storage, s0, s, s_next, timesteps, forces)
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
            parameters = Jutul.setup_parameters(model)
        else
            parameters = p
        end
        dt = step_info[:dt]
        case = JutulCase(model, dt, forces, parameters = parameters, state0 = state0)
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
