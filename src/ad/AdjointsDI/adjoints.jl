import Jutul: AdjointPackedResult

function solve_adjoint_generic(X, F, states, reports_or_timesteps, G;
            # n_objective = nothing,
            extra_timing = false,
            extra_output = false,
            state0 = missing,
            forces = missing,
            info_level = 0,
            kwarg...
        )
        packed_steps = AdjointPackedResult(states, reports_or_timesteps, forces)
        Jutul.set_global_timer!(extra_timing)
        N = length(states)
        n_param = length(X)
        # Allocation part
        if info_level > 1
            jutul_message("Adjoints", "Setting up storage...", color = :blue)
        end
        t_storage = @elapsed storage = setup_adjoint_storage_generic(X, F, packed_steps, G;
            info_level = info_level,
            state0 = state0,
            kwarg...
        )

        if info_level > 1
            jutul_message("Adjoints", "Storage set up in $(Jutul.get_tstr(t_storage)).", color = :blue)
        end
        ∇G = zeros(n_param)

        # Solve!
        if info_level > 1
            jutul_message("Adjoints", "Solving $N adjoint steps...", color = :blue)
        end
        t_solve = @elapsed solve_adjoint_generic!(∇G, X, F, storage, packed_steps, G,
            info_level = info_level,
            state0 = state0,
        )
        if info_level > 1
            jutul_message("Adjoints", "Adjoints solved in $(Jutul.get_tstr(t_solve)).", color = :blue)
        end
        Jutul.print_global_timer(extra_timing; text = "Adjoint solve detailed timing")
        if extra_output
            return (∇G, storage)
        else
            return ∇G
        end
    end

function solve_adjoint_generic!(∇G, X, F, storage, packed_steps::AdjointPackedResult, G;
        info_level = 0,
        state0 = missing
    )
    N = length(packed_steps)
    case = setup_case(X, F, packed_steps, state0, :all)
    Jutul.adjoint_reset_parameters!(storage, case.parameters)

    packed_steps = set_packed_result_dynamic_values!(packed_steps, case)
    H = storage[:adjoint_objective_helper]
    H.packed_steps = packed_steps

    # Do sparsity detection if not already done.
    if info_level > 1
        jutul_message("Adjoints", "Updating sparsity patterns.", color = :blue)
    end

    Jutul.update_objective_sparsity!(storage, G, packed_steps, :forward)
    # Set gradient to zero before solve starts
    @. ∇G = 0
    @tic "sensitivities" for i in N:-1:1
        if info_level > 0
            jutul_message("Step $i/$N", "Solving adjoint system.", color = :blue)
        end
        update_sensitivities_generic!(∇G, X, H, i, G, storage, packed_steps)
    end
    dparam = storage.dparam
    if !isnothing(dparam)
        @. ∇G += dparam
    end
    all(isfinite, ∇G) || error("Adjoint solve resulted in non-finite gradient values.")
    return ∇G
end

function setup_adjoint_storage_generic(X, F, packed_steps::AdjointPackedResult, G;
        state0 = missing,
        backend = missing,
        do_prep = true,
        di_sparse = true,
        info_level = 0,
        single_step_sparsity = true,
        use_sparsity = true
    )
    case = setup_case(X, F, packed_steps, state0, :all)
    packed_steps = set_packed_result_dynamic_values!(packed_steps, case)
    storage = Jutul.setup_adjoint_storage_base(
            case.model, case.state0, case.parameters,
            use_sparsity = use_sparsity,
            linear_solver = Jutul.select_linear_solver(case.model, mode = :adjoint, rtol = 1e-6),
            n_objective = nothing,
            info_level = info_level,
    )
    storage[:dparam] = zeros(length(X))
    setup_jacobian_evaluation!(storage, X, F, G, packed_steps, case, backend, do_prep, single_step_sparsity, di_sparse)
    return storage
end

function set_packed_result_dynamic_values!(packed_steps, case)
    f = case.forces
    if f isa Vector
        steps = map(x -> x[:step], packed_steps.step_infos)
        forces = f[steps]
    else
        forces = [f for _ in 1:length(packed_steps)]
    end
    packed_steps.forces = forces
    packed_steps.state0 = case.state0
    length(forces) == length(packed_steps.step_infos) || error("Expected $(length(packed_steps.step_infos)) forces, got $(length(forces)).")
    return packed_steps
end

function update_sensitivities_generic!(∇G, X, H, i, G, adjoint_storage, packed_steps::AdjointPackedResult)
    state0, state, state_next = Jutul.adjoint_step_state_triplet(packed_steps, i)
    step_info = packed_steps.step_infos[i]
    current_time = step_info[:time]
    report_step = step_info[:step]
    for skey in [:backward, :forward]
        s = adjoint_storage[skey]
        Jutul.reset!(Jutul.progress_recorder(s), step = report_step, time = current_time)
    end

    λ = Jutul.next_lagrange_multiplier!(adjoint_storage, i, G, state0, state, state_next, packed_steps)
    @assert all(isfinite, λ) "Non-finite lagrange multiplier found in step $i. Linear solver failure?"

    H.step_index = i
    prep = adjoint_storage[:prep_di]
    backend = adjoint_storage[:backend_di]
    if isnothing(prep)
        jac = jacobian(H, backend, X)
    else
        jac = jacobian(H, prep, backend, X)
    end
    # jac = jacobian(H, AutoForwardDiff(), X)
    # Add zero entry (corresponding to objective values) to avoid resizing matrix.
    N = length(λ)
    push!(λ, 0.0)
    # tmp = jac'*λ
    mul!(∇G, jac', λ, 1.0, 1.0)
    resize!(λ, N)
    # Gradient of partial objective to parameters
    dparam = adjoint_storage.dparam
    if i == length(packed_steps)
        @. dparam = 0
    end
    dobj = jac[end, :]
    @. dparam += dobj
    return ∇G
end

function setup_case(x::AbstractVector, F, packed_steps::AdjointPackedResult, state0, i::Union{Symbol, Int})
    if i isa Symbol
        i == :all || error("Step index `i` must be positive or `:all`. Got $i.")
        all = true
        i = 1
    else
        all = false
        i > 0 || error("Step index `i` must be positive or `:all`. Got $i.")
        i <= length(packed_steps) || error("Step index `i` must be less than or equal to the number of steps. Got $i, but there are only $(length(packed_steps)) steps.")
    end
    # F(X, step_info) -> model*
    # F(X, step_info) -> model, parameters*
    # F(X, step_info) -> model, parameters, forces*
    # F(X, step_info) -> model, parameters, forces*, state0*
    # F(X, step_info) -> case (current step)
    # F(X, step_info) -> case (all steps)
    # *state0 needs to be provided
    packed_step = packed_steps[i]
    step_info = packed_step.step_info
    N = maximum(x -> x[:step], packed_steps.step_infos)
    c = unpack_setup(step_info, N, F(x, step_info), all = all)
    if c isa JutulCase
        case = c
    else
        model, p, f, s0 = c
        if ismissing(state0)
            state0 = s0
        end
        if ismissing(f)
            f = packed_step.forces
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

mutable struct AdjointObjectiveHelper
    F
    G
    step_index::Union{Int, Missing}
    packed_steps::AdjointPackedResult
    cache::Dict{Tuple{DataType, Int64}, Any}
    function AdjointObjectiveHelper(F, G, packed_steps::AdjointPackedResult)
        new(F, G, missing, packed_steps, Dict())
    end
end

function (H::AdjointObjectiveHelper)(x)
    packed = H.packed_steps
    function evaluate(x, ix)
        s0, s, = Jutul.adjoint_step_state_triplet(packed, ix)
        evaluate_residual_and_jacobian_for_state_pair(x, s, s0, H.F, H.G, packed, ix, H.cache)
    end
    if ismissing(H.step_index)
        # Loop over all to get the "extended sparsity". This is a bit of a hack,
        # but it covers the case where there is some change in dynamics/controls
        # at a later step.
        v = evaluate(x, 1)
        @. v = abs.(v)
        for i in 2:length(packed)
            tmp = evaluate(x, i)
            @. v += abs.(tmp)
        end
    else
        v = evaluate(x, H.step_index)
    end
    return v
end

function setup_jacobian_evaluation!(storage, X, F, G, packed_steps, case0, backend, do_prep, single_step_sparsity, di_sparse)
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

    H = AdjointObjectiveHelper(F, G, packed_steps)
    storage[:adjoint_objective_helper] = H
    # Note: strict = false is needed because we create another function on the fly
    # that essentially calls the same function.
    if do_prep
        if single_step_sparsity
            H.step_index = 1
        else
            H.step_index = missing
        end
        prep = prepare_jacobian(H, backend, X)
    else
        prep = nothing
    end
    storage[:prep_di] = prep
    storage[:backend_di] = backend
    return storage
end

function evaluate_residual_and_jacobian_for_state_pair(x, state, state0, F, G, packed_steps::AdjointPackedResult, step_index::Int, cache = missing)
    step_info = packed_steps[step_index].step_info
    dt = step_info[:dt]
    case = setup_case(x, F, packed_steps, state0, step_index)
    case = reset_context_and_groups(case)
    if step_info[:step] == 1
        state0 = case.state0
    end
    sim = HelperSimulator(case, eltype(x), cache = cache, n_extra = 1)
    forces_for_eval = case.forces
    if forces_for_eval isa AbstractVector
        forces_for_eval = only(forces_for_eval)
    end
    model_residual(state, state0, sim,
        forces = forces_for_eval,
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
    r[end] = G(case.model, s, dt, step_info, forces_for_eval)
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
