function optimize(dopt::DictParameters, objective, setup_fn = dopt.setup_function;
        grad_tol = 1e-6,
        obj_change_tol = 1e-6,
        max_it = 25,
        opt_fun = missing,
        minimize = true,
        simulator = missing,
        config = missing,
        solution_history = false,
        backend_arg = (
            use_sparsity = false,
            di_sparse = true,
            single_step_sparsity = false,
            do_prep = true,
        ),
        kwarg...
    )
    if ismissing(setup_fn)
        error("Setup function was not found in DictParameters struct or as last positional argument.")
    end
    x0, x_setup, limits = optimization_setup(dopt)

    ub = limits.max
    lb = limits.min
    # Set up a cache for forward/backward sim
    adj_cache = setup_optimization_cache(dopt, simulator = simulator, config = config)

    if solution_history
        sols = []
    else
        sols = missing
    end
    solve_and_differentiate(x) = solve_and_differentiate_for_optimization(x, dopt, setup_fn, objective, x_setup, adj_cache;
        backend_arg,
        solution_history = sols
    )
    if dopt.verbose
        jutul_message("Optimization", "Starting calibration of $(length(x0)) parameters.", color = :green)
    end
    t_opt = @elapsed if ismissing(opt_fun)
        v, x, history = Jutul.LBFGS.box_bfgs(x0, solve_and_differentiate, lb, ub;
            maximize = false,
            print = Int(dopt.verbose),
            max_it = max_it,
            grad_tol = grad_tol,
            obj_change_tol = obj_change_tol,
            kwarg...
        )
    else
        self_cache = Dict()
        function f!(x)
            f, g = solve_and_differentiate(x)
            self_cache[:f] = f
            self_cache[:g] = g
            self_cache[:x] = x
            return f
        end

        function g!(z, x)
            if self_cache[:x] !== x
                f!(x)  # Update the cache if the vector has changed
            end
            g = self_cache[:g]
            return z .= g
        end
        x, history = opt_fun(f!, g!, x0, lb, ub)
    end
    if dopt.verbose
        jutul_message("Optimization", "Finished in $t_opt seconds.", color = :green)
    end
    # Also remove AD from the internal ones and update them
    prm_out = deepcopy(dopt.parameters)
    Jutul.AdjointsDI.devectorize_nested!(prm_out, x, x_setup)
    dopt.parameters_optimized = prm_out
    dopt.history = history
    if solution_history
        dopt.history = (history = history, solutions = sols)
    else
        dopt.history = history
    end
    return prm_out
end

function parameters_gradient(dopt::DictParameters, objective, setup_fn = dopt.setup_function;
        simulator = missing,
        config = missing,
        cache = missing,
        raw_output = false,
        output_cache = false,
        backend_arg = (
            use_sparsity = false,
            di_sparse = true,
            single_step_sparsity = false,
            do_prep = true,
        )
    )
    x0, x_setup, = optimization_setup(dopt, include_limits = false)
    if ismissing(cache)
        cache = setup_optimization_cache(dopt, simulator = simulator, config = config)
    end

    f, g = solve_and_differentiate_for_optimization(x0, dopt, setup_fn, objective, x_setup, cache;
        backend_arg
    )
    if raw_output
        if output_cache
            out = (f, g, cache)
        else
            out = (f, g)
        end
    else
        # Put gradients into the same structure as the input
        out = Jutul.AdjointsDI.devectorize_nested(g, x_setup)
        if output_cache
            out = (out, cache)
        end
    end
    return out
end

function freeze_optimization_parameter!(dopt::DictParameters, parameter_name, val = missing)
    parameter_name = convert_key(parameter_name)
    if !ismissing(val)
        set_optimization_parameter!(vc, parameter_name, val)
    end
    delete!(dopt.parameter_targets, parameter_name)
end

function free_optimization_parameter!(dopt::DictParameters, parameter_name;
        initial = missing,
        abs_min = -Inf,
        abs_max = Inf,
        rel_min = -Inf,
        rel_max = Inf
    )
    parameter_name = convert_key(parameter_name)
    if dopt.strict
        if !all(isfinite, rel_max) && !all(isfinite, abs_max)
            throw(ArgumentError("$parameter_name: At least one of the upper bounds (abs_max/rel_max) must be set for free parameters when strict = true"))
        end
        if !all(isfinite, rel_min) && !all(isfinite, abs_min)
            throw(ArgumentError("$parameter_name: At least one of the loewr bounds (abs_min/rel_min) must be set for free parameters when strict = true"))
        end
    end
    if !ismissing(initial)
        set_calibration_parameter!(dopt, parameter_name, initial)
    end
    initial = get_nested_dict_value(dopt.parameters, parameter_name)
    if eltype(initial) isa dopt.active_type
        error("$parameter_name is not an array or single value of the designated active type $(dopt.active_type)")
    end
    check_limit(parameter_name, initial, abs_min, is_max = false, is_rel = false)
    check_limit(parameter_name, initial, abs_max, is_max = true, is_rel = false)
    check_limit(parameter_name, initial, rel_min, is_max = false, is_rel = true)
    check_limit(parameter_name, initial, rel_max, is_max = true, is_rel = true)
    check_limit_pair(parameter_name, initial, rel_min, rel_max, is_rel = true)
    check_limit_pair(parameter_name, initial, abs_min, abs_max, is_rel = false)

    targets = dopt.parameter_targets
    if haskey(targets, parameter_name) && dopt.verbose
        jutul_message("Optimization", "Overwriting limits for $parameter_name.")
    end
    targets[parameter_name] = KeyLimits(rel_min, rel_max, abs_min, abs_max)
    return dopt
end

function free_optimization_parameters!(dopt::DictParameters, targets = all_keys(dopt); kwarg...)
    for k in targets
        free_optimization_parameter!(dopt, k; kwarg...)
    end
    return dopt
end

function set_optimization_parameter!(dopt::DictParameters, parameter_name, value)
    set_nested_dict_value!(dopt.parameters, parameter_name, value)
end
