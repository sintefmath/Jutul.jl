
function solve_and_differentiate_for_optimization(x, dopt::DictParameters, setup_fn, objective, x_setup, adj_cache;
        backend_arg = NamedTuple(),
        gradient = true,
        solution_history = missing
    )

    prm = adj_cache[:parameters]
    function setup_from_vector(X, step_info)
        optimizer_devectorize!(prm, X, x_setup)
        # Return the case setup function This is a function that sets up the
        # case from the parameters
        F() = setup_fn(prm, step_info)
        return redirect_stdout(F, devnull)
    end

    case = setup_from_vector(x, missing)
    objective = Jutul.adjoint_wrap_objective(objective, case.model)
    result = forward_simulate_for_optimization(case, adj_cache)
    packed_steps = Jutul.AdjointPackedResult(result, case)
    packed_steps = Jutul.AdjointsDI.set_packed_result_dynamic_values!(packed_steps, case)

    # Evaluate the objective function
    f = Jutul.evaluate_objective(objective, case.model, packed_steps)
    # Solve adjoints
    if gradient
        if !ismissing(solution_history)
            push!(solution_history, (x = x, states = deepcopy(states), objective = f))
        end
        S = get(adj_cache, :storage, missing)
        if ismissing(S)
            if dopt.verbose
                jutul_message("Optimization", "Setting up adjoint storage.")
            end
            t_setup = @elapsed S = Jutul.AdjointsDI.setup_adjoint_storage_generic(
                x, setup_from_vector, packed_steps, objective;
                backend_arg...,
                info_level = adj_cache[:info_level]
            )
            jutul_message("Optimization", "Finished setup in $t_setup seconds.", color = :green)
            adj_cache[:storage] = S
        end
        g = get(adj_cache, :dx, missing)
        if ismissing(g)
            g = similar(x)
            adj_cache[:dx] = g
        end
        Jutul.AdjointsDI.solve_adjoint_generic!(
            g, x, setup_from_vector, S, packed_steps, objective
        )
    else
        g = missing
    end
    return (f, g)
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

function optimizer_devectorize!(prm, X, x_setup)
    if haskey(x_setup, :lumping) || haskey(x_setup, :scalers)
        X_new = similar(X, 0)
        sizehint!(X_new, length(X))
        pos = 0
        for (i, k) in enumerate(x_setup.names)
            scaler = get(x_setup.scalers, k, missing)
            if haskey(x_setup.lumping, k)
                L = x_setup.lumping[k]
                N = optimizer_devectorize_lumping!(X_new, X, pos, L, scaler)
            else
                N = optimizer_devectorize_scaler!(X_new, X, i, pos, x_setup.offsets, scaler)
            end
            pos += N
        end
        X = X_new
    end
    @assert length(X) == x_setup.offsets[end]-1
    # Set the parameters from the vector
    return Jutul.AdjointsDI.devectorize_nested!(prm, X, x_setup)
end

function optimizer_devectorize_lumping!(X_new, X, pos, L, scaler)
    first_index = L.first_index
    N = length(first_index)
    for v in L.lumping
        push!(X_new, undo_scaler(X[pos + v], scaler))
    end
    return N
end

function optimizer_devectorize_scaler!(X_new, X, i, pos, offsets, scaler)
    N = offsets[i+1]-offsets[i]
    for ix in pos+1:pos+N
        push!(X_new, undo_scaler(X[ix], scaler))
    end
    return N
end

function optimization_setup(dopt::DictParameters; include_limits = true)
    x0, x_setup = Jutul.AdjointsDI.vectorize_nested(dopt.parameters,
        active = active_keys(dopt),
        active_type = dopt.active_type
    )
    length(x0) > 0 || error("Cannot optimize/differentiate zero active parameters. Call free_optimization_parameter! first.")
    if include_limits
        lims = realize_limits(dopt, x_setup)
    else
        lims = missing
    end

    lumping = get_lumping_for_vectorize_nested(dopt)
    scalers = get_scaler_for_vectorize_nested(dopt)
    if length(keys(lumping)) > 0 || length(keys(scalers)) > 0
        off = x_setup.offsets
        x0_new = similar(x0, 0)
        function push_new!(xs, sf)
            for xi in xs
                push!(x0_new, apply_scaler(xi, sf))
            end
        end
        pos = 0
        for (i, k) in enumerate(x_setup.names)
            x_sub = view(x0, off[i]:(off[i+1]-1))
            if haskey(lumping, k)
                x_sub = x_sub[lumping[k].first_index]
            end
            scale = get(scalers, k, missing)
            push_new!(x_sub, scale)
            if include_limits && !ismissing(scale)
                for index in (pos+1):(pos+length(x_sub))
                    lims.min[index] = apply_scaler(lims.min[index], scale)
                    lims.max[index] = apply_scaler(lims.max[index], scale)
                end
            end
            pos += length(x_sub)
        end
        x0 = x0_new
        x_setup = (
            offsets = x_setup.offsets,
            names = x_setup.names,
            dims = x_setup.dims,
            scalers = scalers,
            lumping = lumping
        )
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
