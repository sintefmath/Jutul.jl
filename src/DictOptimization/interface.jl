"""
    optimized_dict = optimize(dopt, objective)
    optimize(dopt::DictParameters, objective, setup_fn = dopt.setup_function;
        grad_tol = 1e-6,
        obj_change_tol = 1e-6,
        max_it = 25,
        opt_fun = missing,
        maximize = false,
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

Optimize parameters defined in a [`DictParameters`](@ref) object using the
provided objective function. At least one variable has to be declared to be free
using `free_optimization_parameter!` prior to calling the optimizer.

# Arguments
- `dopt::DictParameters`: Container with parameters to optimize
- `objective`: The objective function to minimize (or maximize)
- `setup_fn`: Function to set up the optimization problem. Defaults to `dopt.setup_function`

# Keyword Arguments
- `grad_tol`: Gradient tolerance for stopping criterion
- `obj_change_tol`: Objective function change tolerance for stopping criterion
- `max_it`: Maximum number of iterations
- `opt_fun`: Optional custom optimization function. If missing, L-BFGS will be used
- `maximize`: Set to `true` to maximize the objective instead of minimizing
- `simulator`: Optional simulator object used in forward simulations
- `config`: Optional configuration for the setup
- `solution_history`: If `true`, stores all intermediate solutions
- `deps`: One of `:case`, `:parameters`, `:parameters_and_state0`. Defines the
  dependencies for the adjoint computation. See notes for more details.
- `backend_arg`: Options for the autodiff backend:
  - `use_sparsity`: Enable sparsity detection for the objective function
  - `di_sparse`: Use sparse differentiation
  - `single_step_sparsity`: Enable single step sparsity detection (if sparsity
    does not change during timesteps)
  - `do_prep`: Perform preparation step

# Returns
The optimized parameters as a dictionary.

# Notes
- The function stores the optimization history and optimized parameters in the input `dopt` object.
- If `solution_history` is `true`, intermediate solutions are stored in `dopt.history.solutions`.
- The default optimization algorithm is L-BFGS with box constraints.

## Type of dependencies in `deps`
The `deps` argument is used to set the type of dependency the `case` setup
function has on the active optimization parameters. The default, `:case`, is
fully general and allows dependencies on everything contained within the `case`
instance. This can be slow, however, as the setup function must be called for
every time-step. If you know that the model instance and forces are independent
of the active parameters, you can use `deps = :parameters_and_state0`. If there
is no dependence on `state0`, you can set `deps = :parameters`. This can
substantially speed up the optimization process, but as there is no programmatic
verification that this assumption is true, it should be used with care.

This interface is dependent on the model supporting use of
`vectorize_variables!` and `devectorize_variables!` for `state0/parameters`,
which should be the case for most Jutul models.
"""
function optimize(dopt::DictParameters, objective, setup_fn = dopt.setup_function;
        grad_tol = 1e-6,
        obj_change_tol = 1e-6,
        max_it = 25,
        opt_fun = missing,
        maximize = false,
        backend_arg = missing,
        info_level = 0,
        deps::Symbol = :case,
        deps_ad = :jutul,
        simulator = missing,
        config = missing,
        solution_history = false,
        kwarg...
    )
    if ismissing(setup_fn)
        error("Setup function was not found in DictParameters struct or as last positional argument.")
    end
    problem = JutulOptimizationProblem(dopt, objective, setup_fn;
        simulator = simulator,
        config = config,
        info_level = info_level,
        backend_arg = backend_arg,
        solution_history = solution_history,
        deps = deps,
        deps_ad = deps_ad
    )

    if dopt.verbose
        jutul_message("Optimization", "Starting calibration of $(length(problem.x0)) parameters.", color = :green)
    end

    t_opt = @elapsed if ismissing(opt_fun)
        v, x, history = Jutul.LBFGS.box_bfgs(problem;
            print = Int(dopt.verbose),
            max_it = max_it,
            grad_tol = grad_tol,
            obj_change_tol = obj_change_tol,
            maximize = maximize,
            kwarg...
        )
    else
        self_cache = Dict()
        function f!(x)
            f, g = opt_cache(x)
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
    prm_out = optimizer_devectorize(problem, x)
    dopt.parameters_optimized = prm_out
    dopt.history = history
    if solution_history
        dopt.history = (history = history, solutions = problem.solution_history)
    else
        dopt.history = history
    end
    return prm_out
end

"""
    parameters_gradient(dopt::DictParameters, objective, setup_fn = dopt.setup_function)

Compute the gradient of the objective function with respect to the parameters
defined in the `DictParameters` object. This function will return the gradient
as a dictionary with the same structure as the input parameters, where each
entry is a vector of gradients for each parameter. Only gradients with respect
to free parameters will be computed.
"""
function parameters_gradient(dopt::DictParameters, objective, setup_fn = dopt.setup_function;
        simulator = missing,
        config = missing,
        cache = missing,
        raw_output = false,
        output_cache = false,
        deps = :case,
        backend_arg = (
            use_sparsity = false,
            di_sparse = true,
            single_step_sparsity = deps != :case,
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

"""
    freeze_optimization_parameter!(dopt, "parameter_name")
    freeze_optimization_parameter!(dopt, ["dict_name", "parameter_name"])
    freeze_optimization_parameter!(dopt::DictParameters, parameter_name, val = missing)

Freeze an optimization parameter in the `DictParameters` object. This will
remove the parameter from the optimization targets and set its value to `val` if
provided. Any limits/lumping/scaling settings for this parameter will be
removed.
"""
function freeze_optimization_parameter!(dopt::DictParameters, parameter_name, val = missing)
    parameter_name = convert_key(parameter_name)
    if !ismissing(val)
        set_optimization_parameter!(vc, parameter_name, val)
    end
    delete!(dopt.parameter_targets, parameter_name)
end

"""
    free_optimization_parameter!(dopt, "parameter_name", rel_min = 0.01, rel_max = 100.0)
    free_optimization_parameter!(dopt, ["dict_name", "parameter_name"], abs_min = -8.0, abs_max = 7.0)

Free an existing parameter for optimization in the `DictParameters` object. This
will allow the parameter to be optimized through a call to [`optimize`](@ref).

# Nesting structures
If your `DictParameters` has a nesting structure, you can use a vector of
strings or symbols to specify the parameter name, e.g. `["dict_name",
"parameter_name"]` to access the parameter located at
`["dict_name"]["parameter_name"]`.

# Setting limits
The limits can be set using the following keyword arguments:
- `abs_min`: Absolute minimum value for the parameter.  If not set, no absolute
  minimum will be applied.
- `abs_max`: Absolute maximum value for the parameter. If not set, no absolute
  maximum will be applied.
- `rel_min`: Relative minimum value for the parameter. If not set, no relative
  minimum will be applied.
- `rel_max`: Relative maximum value for the parameter. If not set, no relative
  maximum will be applied.

For either of these entries it is possible to pass either a scalar, or an array.
If an array is passed, it must have the same size as the parameter being set.

Note that if `dopt.strict` is set to `true`, at least one of the upper or lower
bounds must be set for free parameters. If `dopt.strict` is set to `false`, the
bounds are optional and the `DictParameters` object can be used to compute
sensitivities, but the built-in optimization routine assumes that finite limits
are set for all parameters.

# Other keyword arguments
- `initial`: Initial value for the parameter. If not set, the current value in
  `dopt.parameters` will be used.
- `scaler=missing`: Optional scaler for the parameter. If not set, no scaling
  will be applied. Available scalers are `:log`, `:exp`. The scaler will be
  applied
- `lumping=missing`: Optional lumping array for the parameter. If not set, no
  lumping will be applied. The lumping array should have the same size as the
  parameter and contain positive integers. The lumping array defines groups of
  indices that should be lumped together, i.e. the same value will be used for
  all indices in the same group. The lumping array should contain all integers
  from 1 to the maximum value in the array, and all indices in the same group
  should have the same value in the initial parameter, otherwise an error will
  be thrown.
"""
function free_optimization_parameter!(dopt::DictParameters, parameter_name;
        initial = missing,
        abs_min = -Inf,
        abs_max = Inf,
        rel_min = -Inf,
        rel_max = Inf,
        scaler = missing,
        lumping = missing
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
        set_optimization_parameter!(dopt, parameter_name, initial)
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
    lumping = validate_and_normalize_lumping(lumping, initial)
    targets = dopt.parameter_targets
    if haskey(targets, parameter_name) && dopt.verbose
        jutul_message("Optimization", "Overwriting limits for $parameter_name.")
    end
    targets[parameter_name] = KeyLimits(rel_min, rel_max, abs_min, abs_max, scaler, lumping)
    return dopt
end

function validate_and_normalize_lumping(lumping::Missing, initial)
    return lumping
end

function validate_and_normalize_lumping(lumping::Bool, initial)
    if lumping
        sz = size(initial)
        lumping = ones(Int, sz)
    end
    return lumping
end

function validate_and_normalize_lumping(lumping, initial)
    if !ismissing(lumping)
        size(lumping) == size(initial) || error("Lumping array must have the same size as the parameter $parameter_name.")
        eltype(lumping) == Int || error("Lumping array must be of type Int.")
        minimum(lumping) >= 1 || error("Lumping array must have positive integers.")
        max_lumping = maximum(lumping)
        for i in 1:max_lumping
            subset = findall(isequal(i), lumping)
            if length(subset) == 0
                error("Lumping array must contain all integers from 1 to $max_lumping.")
            else
                firstval = initial[subset[1]]
                for j in subset
                    if initial[j] != firstval
                        error("Lumping array must contain the same value for all indices in the lumping group $i (value at $j differend from first value at $(subset[1])).")
                    end
                end
            end
        end
    end
    return lumping
end

function free_optimization_parameters!(dopt::DictParameters, targets = all_keys(dopt); kwarg...)
    for k in targets
        free_optimization_parameter!(dopt, k; kwarg...)
    end
    return dopt
end

"""
    set_optimization_parameter!(dopt::DictParameters, parameter_name, value)

Set a specific optimization parameter in the `DictParameters` object. This
function will update the value of the parameter in the `dopt.parameters` dictionary.
"""
function set_optimization_parameter!(dopt::DictParameters, parameter_name, value)
    set_nested_dict_value!(dopt.parameters, parameter_name, value)
end

function add_optimization_multiplier!(dprm::DictParameters, targets...;
        initial = missing,
        lumping = missing,
        name = missing,
        abs_min = -Inf,
        abs_max = Inf
    )
    length(targets) > 0 || error("At least one target parameter must be provided for multiplier.")
    targets = map(t -> convert_key(t), targets)
    if ismissing(name)
        nmult = length(keys(dprm.multipliers))
        name = "multiplier_$(nmult+1)"
    end
    if haskey(dprm.multipliers, name)
        @warn "Multiplier with name $name already exists, overwriting."
    end
    sz = missing
    for t in targets
        t_val = get_nested_dict_value(dprm.parameters, t)
        @info "??" t_val size(t_val)
        sz_t = size(t_val)
        if ismissing(sz)
            sz = sz_t
        else
            sz == sz_t || error("All target parameters must have the same size.")
        end
    end
    if ismissing(initial)
        initial = ones(sz)
    elseif isa(initial, Number)
        initial = fill(initial, sz)
    else
        size(initial) == size(sz) || error("Initial value must have the same size as the target parameters ($sz).")
    end
    lumping = validate_and_normalize_lumping(lumping, initial)
    dprm.multipliers[name] = OptimizationMultiplier(abs_min, abs_max, collect(targets), lumping, initial, copy(initial))
    return dprm
end
