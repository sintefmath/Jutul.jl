function solve_adjoint_generic(X, F, states, reports_or_timesteps, G;
            # n_objective = nothing,
            extra_timing = false,
            extra_output = false,
            state0 = missing,
            forces = missing,
            info_level = 0,
            step_index = eachindex(states),
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
            state0 = state0,
            forces = forces,
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
        t_solve = @elapsed solve_adjoint_generic!(∇G, X, F, storage, states, timesteps, G,
            info_level = info_level,
            state0 = state0,
            forces = forces,
            step_index = step_index
        )
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

function solve_adjoint_generic!(∇G, X, F, storage, states, timesteps, G;
        info_level = 0,
        step_index = eachindex(states),
        state0 = missing,
        forces = missing
    )

    H = storage[:callable_di]
    N = length(timesteps)
    @assert N == length(states)
    badstate0 = ismissing(state0)
    badforces = ismissing(forces)
    if badstate0 || badforces
        sinfo0 = Jutul.optimization_step_info(1, 0.0, timesteps[1], Nstep = N, total_time = sum(timesteps))
        case0 = setup_case(X, F, sinfo0, state0, forces, N; all = true)
        if badstate0
            state0 = case0.state0
        end
        if badforces
            forces = case0.forces
        end
    end
    if forces isa Vector
        forces = forces[step_index]
        @assert length(forces) == N "Expected $N forces (one per time-step), got $(length(forces))."
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
        update_sensitivities_generic!(∇G, X, H, i, G, storage, s0, s, s_next, timesteps, forces)
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
    total_time = sum(timesteps)
    sinfo0 = Jutul.optimization_step_info(1, 0.0, timesteps[1], Nstep = N, total_time = total_time)
    case0 = setup_case(X, F, sinfo0, state0, forces, N)
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
    setup_jacobian_evaluation!(storage, X, F, G, states, case0, forces, timesteps, backend, do_prep, single_step_sparsity, di_sparse)
    return storage
end

function update_sensitivities_generic!(∇G, X, H, i, G, adjoint_storage, state0, state, state_next, timesteps, all_forces)
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
    set_to_step!(H, state, state0, step_info, dt)
    prep = adjoint_storage[:prep_di]
    backend = adjoint_storage[:backend_di]
    if isnothing(prep)
        jac = jacobian(H, backend, X)
    else
        jac = jacobian(H, prep, backend, X)
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

function setup_case(x::AbstractVector, F, step_info, state0, forces, N; all = false)
    # F(X, step_info) -> model*
    # F(X, step_info) -> model, parameters*
    # F(X, step_info) -> model, parameters, forces*
    # F(X, step_info) -> model, parameters, forces*, state0*
    # F(X, step_info) -> case (current step)
    # F(X, step_info) -> case (all steps)
    # *state0 needs to be provided
    c = unpack_setup(step_info, N, F(x, step_info), all = all)
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

function unpack_setup(step_info, N, case::JutulCase; all = false)
    # Either case for all steps or case for current step
    Ns = length(case.dt)
    if Ns > 1
        if case.forces isa Vector
            Ns == N || error("case.forces was a vector and expected $N steps, got $Ns.")
        end
        if !all
            time = step_info[:time]
            pos = length(case.dt)
            t = 0.0
            for (i, dt) in enumerate(case.dt)
                if time >= t && time <= t + dt
                    pos = i
                    break
                end
                t += dt
            end
            case = case[pos]
        end
    end
    return case
end

function unpack_setup(step_info, N, out::Tuple; all = false)
    unpack_setup(step_info, N, out...; all = all)
end

function unpack_setup(step_info, N, model::Jutul.JutulModel; all = false)
    return (model, missing, missing, missing)
end

function unpack_setup(step_info, N, model::Jutul.JutulModel, parameters; all = false)
    return (model, parameters, missing, missing)
end

function unpack_setup(step_info, N, model::Jutul.JutulModel, parameters, forces; all = false)
    return (model, parameters, forces, missing)
end

function unpack_setup(step_info, N, model::Jutul.JutulModel, parameters, forces, state0; all = false)
    return (model, parameters, forces, state0)
end

Base.@kwdef mutable struct AdjointsObjectiveHelper
    state = missing
    state0 = missing
    step_info = missing
    dt = missing
    F
    G
    forces
    N
    cache = Dict()
    states = missing
    timesteps = missing
    case = missing
end

function set_to_step!(H::AdjointsObjectiveHelper, state, state0, step_info, dt)
    # Set the state and state0 to the step
    H.state = state
    H.state0 = state0
    H.step_info = step_info
    H.dt = dt
    H.states = missing
    H.timesteps = missing
    return H
end

function (H::AdjointsObjectiveHelper)(x)
    (; state, state0, step_info, dt, F, G, forces, N, cache, states) = H
    getv(x, s, s0, si, dt_i) = evaluate_residual_and_jacobian_for_state_pair(x, s, s0, si, dt_i, F, G, forces, N, cache)
    if ismissing(states)
        v = getv(x, state, state0, step_info, dt)
    else
        case = H.case
        # Loop over all to get the "extended sparsity".
        # This is a bit of a hack, but it covers the case where there is some change in dynamics/controls at a later step.
        t = 0.0
        # timesteps = case.dt
        timesteps = H.timesteps
        dt_i = timesteps[1]
        total_time = sum(timesteps)
        N = length(timesteps)
        info = Jutul.optimization_step_info(1, t, dt_i, total_time = total_time, Nstep = N)
        v = getv(x, states[1], case.state0, info, dt_i)
        t += dt_i
        @. v = abs.(v)
        for i in 2:N
            dt_i = timesteps[i]
            step_info = Jutul.optimization_step_info(i, t, dt_i, total_time = total_time, Nstep = N)
            tmp = getv(x, states[i], states[i-1], step_info, dt_i)
            @. v += abs.(tmp)
            t += dt_i
        end
    end
    return v
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

    H = AdjointsObjectiveHelper(
        F = F,
        G = G,
        forces = forces,
        N = N,
        case = case0
    )
    storage[:callable_di] = H
    # Note: strict = false is needed because we create another function on the fly
    # that essentially calls the same function.
    if do_prep
        if single_step_sparsity
            H.state = states[1]
            H.state0 = case0.state0
            H.step_info = Jutul.optimization_step_info(1, 0.0, timesteps[1], Nstep = N, total_time = sum(timesteps))
            H.dt = timesteps[1]
            prep = prepare_jacobian(H, backend, X)
        else
            H.states = states
            H.timesteps = timesteps
            prep = prepare_jacobian(H, backend, X)
            H.states = missing
            H.timesteps = missing
        end
        storage[:prep_di] = prep
    else
        storage[:prep_di] = nothing
    end
    storage[:backend_di] = backend
    return storage
end

function evaluate_residual_and_jacobian_for_state_pair(x, state, state0, step_info, dt, F, G, forces, N = 1, cache = missing)
    case = setup_case(x, F, step_info, state0, forces, N)
    case = reset_context_and_groups(case)
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
    s = JutulStorage()
    if sim.model isa Jutul.MultiModel
        for (k, v) in pairs(state)
            if k == :substates
                continue
            end
            s[k] = JutulStorage(v)
        end
    else
        s = JutulStorage(state)
    end
    r[end] = G(case.model, s, dt, step_info, case.forces)
    return copy(r)
end

function reset_context_and_groups(case::Jutul.JutulCase)
    model = reset_context_and_groups(case.model)
    return JutulCase(model, case.dt, case.forces, case.state0, case.parameters, case.input_data)
end

function reset_context_and_groups(model::Jutul.MultiModel{label}) where label
    new_models = Jutul.OrderedDict()
    for (k, m) in pairs(model.models)
        new_models[k] = reset_context_and_groups(m)
    end
    return MultiModel(new_models, label, cross_terms = model.cross_terms)
end

function reset_context_and_groups(model::Jutul.SimulationModel)
    if model.context != Jutul.DefaultContext()
        model = SimulationModel(model.domain, model.system,
            formulation = model.formulation,
            data_domain = model.data_domain,
            extra = model.extra,
            primary_variables = model.primary_variables,
            secondary_variables = model.secondary_variables,
            parameters = model.parameters,
            equations = model.equations
        )
    end
    return model
end
