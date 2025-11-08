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
    n_param_static = storage[:n_static]
    n_param_static == length(X) || error("Internal error: static parameter length mismatch.")
    ∇G = zeros(n_param_static)

    # Solve!
    if info_level > 1
        jutul_message("Adjoints", "Solving $N adjoint steps...", color = :blue)
    end
    solve_adjoint_generic!(∇G, X, F, storage, packed_steps, G,
        info_level = info_level,
        state0 = state0,
    )
    Jutul.print_global_timer(extra_timing; text = "Adjoint solve detailed timing")
    if extra_output
        return (∇G, storage)
    else
        return ∇G
    end
end

function solve_adjoint_generic!(∇G, X, F, storage, states, dt, G; kwarg...)
    packed_steps = AdjointPackedResult(states, dt, missing)
    return solve_adjoint_generic!(∇G, X, F, storage, packed_steps, G; kwarg...)
end

function solve_adjoint_generic!(∇G, X, F, storage, packed_steps::AdjointPackedResult, G;
        info_level = 0,
        state0 = missing
    )
    t_solve = @elapsed begin
        N = length(packed_steps)
        case = setup_case(X, F, packed_steps, state0, :all)
        if F != storage[:F_input]
            @warn "The function F used in the solve must be the same as the one used in the setup."
        end
        # F_dynamic = storage[:F_dynamic]
        F_static = storage[:F_static]
        is_fully_dynamic = storage[:F_fully_dynamic]
        if is_fully_dynamic
            Y = X
            dYdX = missing
        else
            prep = storage[:dF_static_dX_prep]
            backend = storage[:backend_di]
            Y, dYdX = value_and_jacobian(F_static, prep, backend, X)
        end
        G = Jutul.adjoint_wrap_objective(G, case.model)
        Jutul.adjoint_reset_parameters!(storage, case.parameters)

        packed_steps = set_packed_result_dynamic_values!(packed_steps, case)
        H = storage[:adjoint_objective_helper]
        H.num_evals = 0
        H.packed_steps = packed_steps
        dG_dynamic = storage[:dynamic_buffer]
        if storage[:deps_ad] == :jutul
            @assert !is_fully_dynamic "Fully dynamic dependencies must use :di adjoints."
            dG_dynamic_prm = storage[:dynamic_buffer_parameters]
            Jutul.solve_adjoint_sensitivities!(dG_dynamic_prm, storage, packed_steps, G; info_level = 0)
            dstate0 = storage[:dstate0]
            if !ismissing(dstate0)
                dG_dynamic_state0 = storage[:dynamic_buffer_state0]
                dG_dynamic_state0 .= dstate0
            end
        else
            # Do sparsity detection if not already done.
            if info_level > 1
                jutul_message("Adjoints", "Updating sparsity patterns.", color = :blue)
            end

            Jutul.update_objective_sparsity!(storage, G, packed_steps, :forward)
            # Set gradient to zero before solve starts
            @. dG_dynamic = 0
            @tic "sensitivities" for i in N:-1:1
                if info_level > 0
                    jutul_message("Step $i/$N", "Solving adjoint system.", color = :blue)
                end
                update_sensitivities_generic!(dG_dynamic, Y, H, i, G, storage, packed_steps)
            end
            dparam = storage.dparam
            if !isnothing(dparam)
                @. dG_dynamic += dparam
            end
        end

        if is_fully_dynamic
            copyto!(∇G, dG_dynamic)
        else
            mul!(∇G, dYdX', dG_dynamic)
        end
    end
    if info_level > 1
        num_evals = storage[:adjoint_objective_helper].num_evals
        jutul_message("Adjoints", "Adjoints solved in $(Jutul.get_tstr(t_solve)) ($num_evals residual evaluations).", color = :blue)
    end
    all(isfinite, ∇G) || error("Adjoint solve resulted in non-finite gradient values.")
    return ∇G
end

function setup_adjoint_storage_generic(x, F, states, dt, objective; kwarg...)
    packed_steps = AdjointPackedResult(states, dt, missing)
    return setup_adjoint_storage_generic(x, F, packed_steps, objective; kwarg...)
end

function setup_adjoint_storage_generic(X, F, packed_steps::AdjointPackedResult, G;
        state0 = missing,
        do_prep = true,
        di_sparse = true,
        deps = :case,
        deps_ad = :jutul,
        deps_targets = nothing,
        backend = Jutul.default_di_backend(sparse = di_sparse),
        info_level = 0,
        single_step_sparsity = true,
        use_sparsity = true
    )
    case = setup_case(X, F, packed_steps, state0, :all)
    G = Jutul.adjoint_wrap_objective(G, case.model)
    packed_steps = set_packed_result_dynamic_values!(packed_steps, case)
    adj_kwarg = (
        use_sparsity = use_sparsity,
        linear_solver = Jutul.select_linear_solver(case.model, mode = :adjoint, rtol = 1e-6),
        n_objective = nothing,
        info_level = info_level
    )
    if ismissing(backend)
        backend = Jutul.default_di_backend(sparse = di_sparse)
    end
    model = case.model
    if isnothing(deps_targets)
        deps_targets = Jutul.parameter_targets(model)
    end
    # Two approaches:
    # 1. F_static(X) -> Y (vector of parameters) -> F_dynamic(Y) (updated case)
    # 2. F(X) -> case directly and F_dynamic = F and F_static = identity
    # Three different variants:
    # Approach 1 can differentiate using DI or Jutul adjoints for the F_dynamic part
    # Approach 2 can only differentiate using DI adjoints - uses the entire case setup at all steps
    deps in (:case, :parameters, :parameters_and_state0) || error("deps must be :case, :parameters or :parameters_and_state0. Got $deps.")
    prep_static = nothing
    adj_kwarg = (
        use_sparsity = use_sparsity,
        linear_solver = Jutul.select_linear_solver(case.model, mode = :adjoint, rtol = 1e-6),
        n_objective = nothing,
        info_level = info_level,
    )
    use_di = deps_ad == :di
    fully_dynamic = deps == :case
    inc_state0 = deps == :parameters_and_state0
    if fully_dynamic || use_di
        storage = Jutul.setup_adjoint_storage_base(
            case.model, case.state0, case.parameters;
            adj_kwarg...
        )
    else
        storage = Jutul.setup_adjoint_storage(model;
            state0 = case.state0,
            parameters = case.parameters,
            include_state0 = inc_state0,
            targets = deps_targets,
            adj_kwarg...
        )
    end
    # Prepare Jacobian evaluation
    if fully_dynamic
        # If we are in "fully dynamic" mode we always differentiate the entire
        # setup function. This can be costly but covers everything.
        F_dynamic = F
        F_static = x -> x
        deps_ad = :di
        parameter_map = state0_map = missing
        N_prm = length(X)
        N_state0 = 0
    else
        # We are doing a dynamic-static split for the chain rule. Two options:
        # DifferentiationInterface.jl (DI) for both, or DI for static (X to
        # parameters) and Jutul adjoints for dynamic (parameters to case).
        if use_di
            parameter_map, = Jutul.variable_mapper(model, :parameters,
                targets = deps_targets,
                config = nothing
            )
            if inc_state0
                state0_map, = Jutul.variable_mapper(model, :primary)
            else
                state0_map = missing
            end
            prm_model = case.model
        else
            # NOTE: It is important that we use the parameter model here and the
            # corresponding map, as the gradients may otherwise be
            # inconsistently ordered with what we are using inside the adjoint
            # code. We do not really care about the nature of the ordering, just
            # that it is consistent in the code.
            parameter_map = storage.parameter_map
            state0_map = storage.state0_map
            prm_model = storage.parameter.model
        end
        N_prm = Jutul.vectorized_length(case.model, parameter_map)
        if inc_state0
            N_state0 = Jutul.number_of_degrees_of_freedom(case.model)
        else
            N_state0 = 0
        end
        cache_static = Dict{Type, AbstractVector}()
        F_static = (X, step_info = missing) -> map_X_to_Y(F, X, prm_model, parameter_map, state0_map, cache_static)
        F_dynamic = (Y, step_info = missing) -> setup_from_vectorized(Y, case, parameter_map, state0_map; model = prm_model, step_info = step_info)
        if do_prep
            prep_static = prepare_jacobian(F_static, backend, X)
        end
    end
    # Whatever was input - for checking
    storage[:F_input] = F
    # Dynamic part - every timestep
    storage[:F_dynamic] = F_dynamic
    # Static part - once
    storage[:F_static] = F_static
    # Jacobian action
    storage[:dF_static_dX_prep] = prep_static
    storage[:F_fully_dynamic] = fully_dynamic
    # Hints about what type of dependencies are used
    storage[:deps] = deps
    storage[:deps_ad] = deps_ad

    # Switch to Y and F_dynamic(Y) as main function
    Y = F_static(X)
    storage[:n_static] = length(X)
    storage[:n_dynamic] = length(Y)
    # Buffer used for dynamic gradient storage
    dG_dyn = similar(Y)
    storage[:dynamic_buffer] = dG_dyn
    storage[:dynamic_buffer_parameters] = view(dG_dyn, 1:N_prm)
    storage[:dynamic_buffer_state0] = view(dG_dyn, (N_prm+1):(N_prm+N_state0))
    H = AdjointObjectiveHelper(F_dynamic, G, packed_steps)
    storage[:adjoint_objective_helper] = H
    if do_prep
        if single_step_sparsity
            step_index = :firstlast
        else
            step_index = :all
        end
        set_objective_helper_step_index!(H, case.model, step_index)
        prep = prepare_jacobian(H, backend, Y)
    else
        prep = nothing
    end
    storage[:prep_di] = prep
    storage[:backend_di] = backend
    if use_di
        storage[:dparam] = zeros(storage[:n_dynamic])
    else
        # state0 is a bit strange
        storage[:dparam] = zeros(N_prm)
    end

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
    packed_steps.input_data = case.input_data
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
        Jutul.recorder_reset!(Jutul.progress_recorder(s), step = report_step, time = current_time)
    end

    λ = Jutul.next_lagrange_multiplier!(adjoint_storage, i, G, state0, state, state_next, packed_steps)
    @assert all(isfinite, λ) "Non-finite lagrange multiplier found in step $i. Linear solver failure?"

    H.step_index = i
    set_objective_helper_step_index!(H, adjoint_storage.forward.model, i)
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
    N = step_info[:Nstep]
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
            case = case[step_info[:step]]
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
    objective_evaluator
    step_index::Union{Symbol, Int}
    packed_steps::AdjointPackedResult
    cache::Dict{Tuple{DataType, Int64}, Any}
    num_evals::Int64
    function AdjointObjectiveHelper(F, G, packed_steps::AdjointPackedResult)
        new(F, G, missing, :all, packed_steps, Dict(), 0)
    end
end

function set_objective_helper_step_index!(H::AdjointObjectiveHelper, model, step_index)
    if step_index isa Int
        step_index > 0 || error("Step index must be positive. Got $step_index.")
        step_index <= length(H.packed_steps) || error("Step index $step_index is larger than the number of steps $(length(H.packed_steps)).")
    else
        step_index in (:firstlast, :all) || error("If step_index is a Symbol it must be :firstlast or :all.")
        step_index isa Symbol || error("Step index must be an integer or symbol (:firstlast, :all). Got $step_index.")
    end
    H.step_index = step_index
    H.objective_evaluator = Jutul.objective_evaluator_from_model_and_state(H.G, model, H.packed_steps, step_index)
end

function (H::AdjointObjectiveHelper)(x)
    packed = H.packed_steps
    function evaluate(x, ix)
        s0, s, = Jutul.adjoint_step_state_triplet(packed, ix)
        is_sum = H.G isa Jutul.AbstractSumObjective
        H.G::Jutul.AbstractJutulObjective
        H.num_evals += 1
        evaluate_residual_and_jacobian_for_state_pair(x, s, s0, H.F, H.objective_evaluator, packed, ix, H.cache; is_sum = is_sum)
    end
    if H.step_index isa Symbol
        # Loop over multiple steps to get the "extended sparsity". This is a bit
        # of a hack, but it covers the case where there is some change in
        # dynamics/controls at a later step.
        if H.step_index == :all
            indices_to_eval = eachindex(packed.states)
        else
            H.step_index == :firstlast || error("Unknown step index symbol: $(indices_to_eval).")
            indices_to_eval = [1, length(packed.states)]
        end
        v = evaluate(x, indices_to_eval[1])
        @. v = abs.(v)
        for (i, step) in enumerate(indices_to_eval)
            if i == 1
                continue
            end
            tmp = evaluate(x, step)
            @. v += abs.(tmp)
        end
    else
        v = evaluate(x, H.step_index)
    end
    return v
end

function setup_outer_chain_rule(F, case, deps::Symbol)
    return (F_dynamic, F_static, dF_static_dX)
end

function evaluate_residual_and_jacobian_for_state_pair(x, state, state0, F, objective_eval::Function, packed_steps::AdjointPackedResult, substep_index::Int, cache = missing; is_sum = true)
    step_info = packed_steps[substep_index].step_info
    dt = step_info[:dt]
    step_index = step_info[:step]
    if is_sum
        case = setup_case(x, F, packed_steps, state0, substep_index)
    else
        case = setup_case(x, F, packed_steps, state0, :all)
    end
    case = reset_context_and_groups(case)
    if step_info[:step] == 1
        state0 = case.state0
    end
    sim = HelperSimulator(case, eltype(x), cache = cache, n_extra = 1)
    forces_for_eval = case.forces
    forces_is_vec = forces_for_eval isa AbstractVector
    if is_sum
        if forces_is_vec
            forces_for_eval = only(forces_for_eval)
        end
        forces_arg = (forces = forces_for_eval,)
    else
        if forces_is_vec
            allforces = case.forces
            forces_for_eval = forces_for_eval[step_index]
        else
            allforces = [forces_for_eval for _ in 1:step_info[:Nstep]]
        end
        forces_arg = (allforces = allforces, forces = forces_for_eval,)
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
    r[end] = objective_eval(case.model, s;
        input_data = case.input_data,
        parameters = case.parameters,
        forces_arg...
    )
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
