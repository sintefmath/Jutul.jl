
function solve_and_differentiate_for_optimization(x, dopt::DictParameters, setup_fn, objective, x_setup, adj_cache;
        backend_arg = NamedTuple(),
        gradient = true
    )

    prm = adj_cache[:parameters]
    function setup_from_vector(X, step_info)
        # Set the parameters from the vector
        Jutul.AdjointsDI.devectorize_nested!(prm, X, x_setup)
        # Return the case setup function
        # This is a function that sets up the case from the parameters
        F() = setup_fn(prm, step_info)
        return redirect_stdout(F, devnull)
    end

    case = setup_from_vector(x, missing)
    states, dt = forward_simulate_for_optimization(case, adj_cache)
    # Evaluate the objective function
    f = Jutul.evaluate_objective(objective, case.model, states, dt, case.forces)
    # Solve adjoints
    if gradient
        S = get(adj_cache, :storage, missing)
        if ismissing(S)
            if dopt.verbose
                jutul_message("Optimization", "Setting up adjoint storage.")
            end
            t_setup = @elapsed S = Jutul.AdjointsDI.setup_adjoint_storage_generic(
                x, setup_from_vector, states, dt, objective;
                backend_arg...,
                info_level = adj_cache[:config][:info_level]
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
            g, x, setup_from_vector, S, states, dt, objective,
        )
        # g = Jutul.AdjointsDI.solve_adjoint_generic(
        #     x, setup_from_vector, states, dt, objective,
        #     use_sparsity = false,
        #     di_sparse = false,
        #     single_step_sparsity = false,
        #     do_prep = false
        # )
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
        config = simulator_config(sim, info_level = -1, output_substates = true)
        adj_cache[:config] = config
    end
    result = simulate!(sim, case.dt,
        config = config,
        state0 = case.state0,
        parameters = case.parameters,
        forces = case.forces
    )
    states, dt, = Jutul.expand_to_ministeps(result)
    return (states, dt)
end



function optimization_setup(dopt::DictParameters; include_limits = true)
    x0, x_setup = Jutul.AdjointsDI.vectorize_nested(dopt.parameters,
        active = active_keys(dopt),
        active_type = dopt.active_type
    )
    length(x0) > 0 || error("Cannot optimize/differentiate zero active parameters. Call free_optimization_parameter! first.")
    if include_limits
        lims = realize_limits(dopt)
    else
        lims = missing
    end
    return (x0 = x0, x_setup = x_setup, limits = lims)
end

function setup_optimization_cache(dopt::DictParameters; simulator = missing, config = missing)
    # Set up a cache for forward/backward sim
    adj_cache = Dict()
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
