
function solve_and_differentiate_for_optimization(x, dopt::DictParameters, setup_fn, objective, x_setup, adj_cache;
        backend_arg = NamedTuple(),
        gradient = true,
        solution_history = missing,
        print_parameters = false
    )
    prm = adj_cache[:parameters]
    setup_from_vector = (X, step_info = missing) -> setup_from_vector_optimizer(X, step_info, setup_fn, prm, x_setup)
    case = setup_from_vector(x, missing)
    objective = Jutul.adjoint_wrap_objective(objective, case.model)
    if print_parameters
        prm_opt = deepcopy(prm)
        optimizer_devectorize!(prm_opt, x, x_setup)
        dopt.parameters_optimized = prm_opt
        print_optimization_overview(dopt)
    end
    result = forward_simulate_for_optimization(case, adj_cache)
    adj_cache[:last_forward_result] = result
    packed_steps = Jutul.AdjointPackedResult(result, case)
    packed_steps = Jutul.AdjointsDI.set_packed_result_dynamic_values!(packed_steps, case)

    # Evaluate the objective function
    f = Jutul.evaluate_objective(objective, case.model, packed_steps)
    adj_cache[:forward_count] += 1
    # Solve adjoints
    if gradient
        if !ismissing(solution_history)
            push!(solution_history, (x = x, states = deepcopy(states), objective = f))
        end
        S = get(adj_cache, :storage, missing)
        if ismissing(S)
            if dopt.verbose
                jutul_message("Optimization", "Setting up adjoint storage.", color = :green)
            end
            t_setup = @elapsed S = Jutul.AdjointsDI.setup_adjoint_storage_generic(
                x, setup_from_vector, packed_steps, objective;
                backend_arg...,
                info_level = adj_cache[:info_level]
            )
            # Make sure that tolerances match between forward and adjoint configs
            S[:forward_config][:tolerances] = adj_cache[:config][:tolerances]
            if dopt.verbose
                jutul_message("Optimization", "Finished setup in $t_setup seconds.", color = :green)
            end
            adj_cache[:storage] = S
        end
        # Some optimizers use the return value beyond a single call. So we
        # create a new gradient array each time to avoid having this be aliased
        # with a previous output.
        g = similar(x)
        Jutul.AdjointsDI.solve_adjoint_generic!(
            g, x, setup_from_vector, S, packed_steps, objective
        )
        adj_cache[:backward_count] += 1
    else
        g = missing
    end
    if dopt.verbose
        num_f = adj_cache[:forward_count]
        fmt = x -> @sprintf("%2.3e", x)
        if gradient
            dg = sqrt(sum(abs2, g))
            gstr = ", gradient 2-norm: $(fmt(dg))"
        else
            gstr = " (no gradient evaluated)"
        end
        println("")
        jutul_message("Optimization", "Objective #$num_f: $(fmt(f))$gstr", color = :green)
    end
    return (f, g)
end

function setup_from_vector_optimizer(X, step_info, setup_fn, prm, x_setup)
    # Make a nested shallow dict copy? There is a kind of bug here...
    # prm = dict_shallow_copy(prm)
    prm = deepcopy(prm)
    optimizer_devectorize!(prm, X, x_setup)
    # Return the case setup function This is a function that sets up the
    # case from the parameters
    F() = setup_fn(prm, step_info)
    return redirect_stdout(F, devnull)
end

function dict_shallow_copy(x::T) where T<:AbstractDict
    new_x = T()
    for (k, v) in pairs(x)
        new_x[k] = dict_shallow_copy(v)
    end
    return new_x
end

function dict_shallow_copy(x)
    return x
end

function forward_simulate_for_optimization(case, adj_cache)
    sim = get(adj_cache, :simulator, missing)
    if ismissing(sim)
        sim = Jutul.Simulator(case)
        adj_cache[:simulator] = sim
    end
    config = get(adj_cache, :config, missing)
    if ismissing(config)
        config = simulator_config(sim,
            info_level = adj_cache[:info_level],
            end_report = false,
            output_substates = true
        )
        adj_cache[:config] = config
    end
    return simulate!(sim, case.dt,
        config = config,
        state0 = case.state0,
        parameters = case.parameters,
        forces = case.forces
    )
end

function optimizer_devectorize!(prm, X, x_setup; multipliers = missing)
    if haskey(x_setup, :lumping) || haskey(x_setup, :scalers)
        X_new = similar(X, 0)
        sizehint!(X_new, length(X))
        pos = 0
        for (i, k) in enumerate(x_setup.names)
            scaler = get(x_setup.scalers, k, missing)
            lumping = get(x_setup.lumping, k, missing)
            stats = x_setup.statistics[k]
            if ismissing(x_setup.limits)
                minlims = missing
                maxlims = missing
            else
                minlims = x_setup.limits.min
                maxlims = x_setup.limits.max
            end
            N = optimizer_devectorize_scaler!(X_new, X, i, pos, x_setup.offsets, minlims, maxlims, stats, lumping, scaler)
            pos += N
        end
        X = X_new
    end
    @assert length(X) == x_setup.offsets[end]-1
    # Set the parameters from the vector
    return Jutul.AdjointsDI.devectorize_nested!(prm, X, x_setup, multipliers = multipliers)
end

function optimizer_devectorize_scaler!(X_new, X, i, pos, offsets, minlims, maxlims, stats, lumping, scaler)
    if ismissing(lumping)
        N = offsets[i+1]-offsets[i]
        ind = pos+1:pos+N
        lim_bnds = group_limits(minlims, maxlims, ind)
        for (i, ix) in enumerate(ind)
            lim_val = scaler_limits(minlims, maxlims, i)
            bnds = LimitBounds(lim_val, lim_bnds)
            push!(X_new, undo_scaler(X[ix], bnds, stats, scaler))
        end
    else
        first_index = lumping.first_index
        N = length(first_index)
        ind = pos+1:pos+N
        lim_bnds = group_limits(minlims, maxlims, ind)
        for (i, v) in enumerate(lumping.lumping)
            lim_val = scaler_limits(minlims, maxlims, pos + v)
            bnds = LimitBounds(lim_val, lim_bnds)
            push!(X_new, undo_scaler(X[pos + v], bnds, stats, scaler))
        end
    end
    return N
end

function scaler_limits(minlims, maxlims, c)
    if ismissing(minlims)
        min_limit = -Inf
    else
        min_limit = minlims[c]
    end
    if ismissing(maxlims)
        max_limit = Inf
    else
        max_limit = maxlims[c]
    end
    return (min = min_limit, max = max_limit)
end

function group_limits(minlims, maxlims, ind)
    if ismissing(minlims)
        min_limit = -Inf
    else
        min_subv = view(minlims, ind)
        min_limit = minimum(min_subv)
    end
    if ismissing(maxlims)
        max_limit = -Inf
    else
        max_subv = view(maxlims, ind)
        max_limit = maximum(max_subv)
    end
    return (min = min_limit, max = max_limit)
end

function optimization_setup(dopt::DictParameters; include_limits = true)
    x0, x_setup = Jutul.AdjointsDI.vectorize_nested(dopt.parameters,
        multipliers = dopt.multipliers,
        active = active_keys(dopt),
        active_type = dopt.active_type
    )
    length(x0) > 0 || error("Cannot optimize/differentiate zero active parameters. Call free_optimization_parameter! first.")
    if include_limits
        lims = realize_limits(dopt, x_setup)
    else
        lims = (min = missing, max = missing)
    end

    lumping = get_lumping_for_vectorize_nested(dopt)
    scalers = get_scaler_for_vectorize_nested(dopt)
    base_limits = (min = Float64[], max = Float64[])
    if length(keys(lumping)) > 0 || length(keys(scalers)) > 0
        off = x_setup.offsets
        x0_new = similar(x0, 0)
        pos = 0
        for (i, k) in enumerate(x_setup.names)
            stats = x_setup.statistics[k]
            x_sub = view(x0, off[i]:(off[i+1]-1))
            if haskey(lumping, k)
                x_sub = x_sub[lumping[k].first_index]
            end
            scale = get(scalers, k, missing)
            N = length(x_sub)
            lim_bnds = group_limits(lims.min, lims.max, pos+1:pos+N)
            for (j, xi) in enumerate(x_sub)
                index = pos + j
                lim_val = scaler_limits(lims.min, lims.max, index)
                # Store the limits for this variable and the group for use in
                # scaling
                bnds = LimitBounds(lim_val, lim_bnds)
                push!(base_limits.min, bnds.min)
                push!(base_limits.max, bnds.max)
                S(v) = apply_scaler(v, bnds, stats, scale)
                push!(x0_new, S(xi))
                if include_limits && !ismissing(scale)
                    scale_min = S(bnds.min)
                    scale_max = S(bnds.max)
                    if scale_min > scale_max
                        error("Inconsistent limits after scaling for parameter $k at index $index: ($(bnds.min), $(bnds.max)) scaled to ($scale_min, $scale_max) with $scale: Bounds = $bnds")
                    elseif !isfinite(scale_min) || !isfinite(scale_max)
                        error("Non-finite limits after scaling for parameter $k at index $index: ($(bnds.min), $(bnds.max)) scaled to ($scale_min, $scale_max) with $scale: Bounds = $bnds")
                    end
                    lims.min[index] = scale_min
                    lims.max[index] = scale_max
                end
            end
            pos += length(x_sub)
        end
        x0 = x0_new
        old_keys = keys(x_setup)
        x_setup = Jutul.AdjointsDI.vectorize_nested_meta(
            x_setup.offsets,
            x_setup.names,
            x_setup.dims,
            x_setup.types,
            multiplier_targets = x_setup.multiplier_targets,
            statistics = x_setup.statistics,
            scalers = scalers,
            lumping = lumping,
            limits = base_limits
        )
        @assert keys(x_setup) == old_keys "Keys changed during optimization setup: got $(keys(x_setup)), expected $old_keys"
    end
    if include_limits
        @assert length(lims.min) == length(lims.max) "Upper bound length ($(length(lims.max))) does not match lower bound length ($(length(lims.min)))."
        @assert length(lims.max) == length(x0) "Bound length ($(length(lims.max))) does not match parameter vector length ($(length(x0)))."
    end
    return (x0 = x0, x_setup = x_setup, limits = lims)
end

function setup_optimization_cache(dopt::DictParameters;
        simulator = missing,
        config = missing,
        info_level = 0
    )
    # Set up a cache for forward/backward sim
    adj_cache = Dict()
    adj_cache[:forward_count] = 0
    adj_cache[:backward_count] = 0
    adj_cache[:info_level] = info_level
    # Internal copy - to be used for adjoints etc
    adj_cache[:parameters] = widen_dict_copy(dopt.parameters)
    if !ismissing(simulator)
        adj_cache[:simulator] = simulator
    end
    if !ismissing(config)
        adj_cache[:config] = config
    end
    return adj_cache
end
