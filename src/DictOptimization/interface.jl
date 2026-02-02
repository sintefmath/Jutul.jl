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
- `optimizer`: Symbol defining the optimization algorithm to use. Available
  options are `:lbfgs` (default), `:lbfgsb_qp` and `:lbfgsb` (requires LBFGSB.jl
  to be imported)
- `opt_fun`: Optional custom optimization function. If missing, L-BFGS will be
  used. Takes in a NamedTuple containing fields `f`, `g`, `x0`, `min`, `max`.
  Here, `f(x)` returns the objective function value at `x`, `g(dFdx, x)` fills
  `dFdx` with the gradient at `x`, `x0` is the initial guess, and `min` and
  `max` are the lower and upper bounds, respectively. The functions `u =
  F.scale(x)` and `x = F.descale(u)` can be used to convert between scaled and
  unscaled variables. Nominally, the initial values are scaled to the unit cube
  and the solution must thus be unscaled before usage. Gradients and internal
  scaling/descaling is automatically handled.
- `maximize`: Set to `true` to maximize the objective instead of minimizing
- `gradient_scaling`: If `true`, internally scales the objective gradient
  according to the initial 2-norm of the gradient. If a `Float64` value is
  provided, that value is used as a global scaling factor for the gradient. The
  internal gradient and objective value is divided by the chosen scaling.
- `simulator`: Optional simulator object used in forward simulations
- `config`: Optional configuration for the setup
- `solution_history`: If `true`, stores all intermediate solutions
- `deps`: One of `:case`, `:parameters`, `:parameters_and_state0`. Defines the
  dependencies for the adjoint computation. See notes for more details.
- `backend_arg`: Options for the autodiff backend:
  - `use_sparsity`: Enable sparsity detection for the objective function
  - `di_sparse`: Use sparse differentiation
  - `single_step_sparsity`: Enable single step sparsity detection (if sparsity
    does not change during timesteps). This means that the solver will assume
    that the sparsity pattern will be determined entirely by the first and last
    steps of the simulation. Alternatively, this can be set to `:unique_forces`
    to use the sparsity pattern determined by all unique force terms in the
    solve, `:firstlast` to only use the first and last time steps or `:allsteps`
    to use all time steps (the latter is equivalent to setting `use_sparsity` to
    `true`).
  - `do_prep`: Perform preparation step

# Returns
The optimized parameters as a dictionary.

# Notes
- The function stores the optimization history and optimized parameters in the
  input `dopt` object.
- If `solution_history` is `true` or :x, intermediate solutions are stored in
  `dopt.history.solutions`. If it is set to `:full`, the full states are also
  copied and stored for each iteration. This can use a lot of memory for large
  simulations.
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
        optimizer = :lbfgs,
        maximize = false,
        backend_arg = missing,
        info_level = 0,
        deps::Symbol = :case,
        deps_ad = :jutul,
        simulator = missing,
        config = missing,
        solution_history = false,
        print_parameters = false,
        allow_errors = false,
        scale = optimizer == :lbfgsb_qp,
        gradient_scaling = true,
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
        deps_ad = deps_ad,
        print_parameters = print_parameters,
        allow_errors = allow_errors,
        gradient_scaling = gradient_scaling
    )

    if dopt.verbose
        jutul_message("Optimization", "Starting calibration of $(length(problem.x0)) parameters.", color = :green)
    end

    t_opt = @elapsed if ismissing(opt_fun)
        x, solver_history = optimize_implementation(problem, Val(optimizer); 
            grad_tol = grad_tol,
            obj_change_tol = obj_change_tol,
            max_it = max_it,
            maximize = maximize,
            scale = scale,
            kwarg...
        )
    else
        F = Jutul.DictOptimization.setup_optimization_functions(problem, maximize = maximize, scale = scale)
        x = opt_fun(F)
        x = F.descale(x)
        solver_history = F.history
    end
    if dopt.verbose
        jutul_message("Optimization", "Finished in $t_opt seconds.", color = :green)
    end
    # Also remove AD from the internal ones and update them
    prm_out = optimizer_devectorize(problem, x)
    dopt.parameters_optimized = prm_out

    # Store information about the solver progress in the history
    history = Dict()
    history[:objectives] = problem.cache[:objectives]
    history[:gradient_norms] = problem.cache[:gradient_norms]
    history[:solutions] = problem.cache[:solutions]
    history[:solver_history] = solver_history
    dopt.history = NamedTuple(history)

    return prm_out
end

function optimize_implementation(problem, ::Val{:lbfgs}; scale = true, kwarg...)
    if !scale
        error("Standard lbfgs optimization without scaling is not supported.")
    end
    v, x, history = Jutul.LBFGS.box_bfgs(problem;
        kwarg...
    )
    return (x, history)
end

function optimize_implementation(problem, ::Val{:lbfgsb_qp}; kwarg...)
    v, x, history = Jutul.LBFGS.optimize_bound_constrained(problem;
        kwarg...
    )
    return (x, history)
end

function optimize_implementation(problem, ::Val{optimizer}; kwarg...) where optimizer
    error("Unknown optimizer: $optimizer (available: :lbgs, :lbfgsb (requires LBFGSB.jl to be imported))")
end

function setup_optimization_functions(problem::JutulOptimizationProblem; maximize = false, scale = false)
    x0 = problem.x0
    n = length(x0)
    ub = problem.limits.max
    lb = problem.limits.min
    length(lb) == n || throw(ArgumentError("Length of lower bound ($(length(lb))) must match length of initial guess ($n)"))
    length(ub) == n || throw(ArgumentError("Length of upper bound ($(length(ub))) must match length of initial guess ($n)"))
    # Check bounds
    for i in eachindex(x0, lb, ub)
        if lb[i] >= ub[i]
            throw(ArgumentError("Lower bound must be less than upper bound for index $i: lb[$i] = $(lb[i]), ub[$i] = $(ub[i])"))
        end
        if x0[i] < lb[i] || x0[i] > ub[i]
            throw(ArgumentError("Initial guess x0[$i] = $(x0[i]) is outside bounds [$(lb[i]), $(ub[i])]"))
        end
        if !isfinite(lb[i]) || !isfinite(ub[i])
            throw(ArgumentError("Bounds must be finite, got lb[$i] = $(lb[i]), ub[$i] = $(ub[i])"))
        end
    end
    # Use local variables to handle caching
    δ = ub .- lb
    function dx_to_du!(g)
        if scale
            for i in eachindex(g, δ)
                g[i] = g[i] * δ[i]
            end
        end
        return g
    end

    function x_to_u(x)
        if scale
            u = (x - lb) ./ δ
        else
            u = x
        end
        return u
    end

    function u_to_x(u)
        if scale
            x = u .* δ + lb
        else
            x = u
        end
        return x
    end

    prev_hash = NaN
    prev_val = NaN
    prev_grad = similar(ub)
    history = Float64[]
    function F(x, dFdx = missing)
        # Whenever this function is called, we also compute the gradient. We
        # then leave it around for fetching next time if needed by hashing. A
        # potential improvement would be to avoid computing the gradient if not
        # actually needed.
        hash_x = hash(x)
        if prev_hash == hash_x
            obj = prev_val
        else
            if scale
                x = u_to_x(x)
            end
            f, g = problem(x; gradient = true)
            dx_to_du!(g)
            if maximize
                f = -f
                @. g = -g
            end
            prev_val = obj = f
            prev_grad .= g
            prev_hash = hash_x
            push!(history, obj)
        end
        if !ismissing(dFdx)
            dFdx .= prev_grad
        end
        return obj
    end
    # Evaluate objective
    function f!(x)
        return F(x)
    end
    # Objective and gradient
    function g!(dFdx, x)
        F(x, dFdx)
        return dFdx
    end
    if scale
        lb_scaled = zeros(n)
        ub_scaled = ones(n)
    else
        lb_scaled = lb
        ub_scaled = ub
    end
    return (
        f = f!,
        g = g!,
        history = history,
        min = lb_scaled,
        max = ub_scaled,
        x0 = x_to_u(x0),
        scale = x_to_u,
        descale = u_to_x
    )
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
            use_sparsity = true,
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
        backend_arg = backend_arg
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
    parameter_name = convert_key(parameter_name, dopt.parameters)
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
  will be applied. Available scalers are:
    - `:log`: Logarithmic scaling. This value uses shifts to avoid issues with zero
      values.
    - `:exp`: Exponential scaling
    - `:linear`: Linear scaling (scaling to bounds of values, guaranteeing
      values between between 0 and 1 for initial values.)
    - `linear_limits`: Linear scaling with limits (scaling to bounds of values,
      guaranteeing values between between 0 and 1 for all values within the
      limits.)
    - `reciprocal`: Reciprocal scaling
    - `log10`: Base-10 logarithmic scaling
    - `log`: Base-e logarithmic scaling without shifts
    - A custom scaler object implementing the `DictOptimizationScaler` interface.
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
    parameter_name = convert_key(parameter_name, dopt.parameters)
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
    lumping = validate_and_normalize_lumping(lumping, initial, parameter_name)
    targets = dopt.parameter_targets
    if haskey(targets, parameter_name) && dopt.verbose
        jutul_message("Optimization", "Overwriting limits for $parameter_name.")
    end
    targets[parameter_name] = KeyLimits(rel_min, rel_max, abs_min, abs_max, scaler, lumping)
    return dopt
end

function validate_and_normalize_lumping(lumping::Missing, initial, parameter_name)
    return lumping
end

function validate_and_normalize_lumping(lumping::Bool, initial, parameter_name)
    if lumping
        sz = size(initial)
        lumping = ones(Int, sz)
    end
    return lumping
end

function validate_and_normalize_lumping(lumping, initial, parameter_name)
    if !ismissing(lumping)
        szl = size(lumping)
        szi = size(initial)
        szl == szi || error("Lumping array (size $szl) must have the same size as the parameter $parameter_name ($szi).")
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

"""
    add_optimization_multiplier!(dprm::DictParameters, name_of_target; abs_min = 0.2, abs_max = 5.0)
    add_optimization_multiplier!(dprm, target1, target2, target3; abs_min = 0.2, abs_max = 5.0, initial = 2.0)

Add an optimization multiplier that acts on one or more targets to the
`DictParameters` object. The multiplier will be optimized during the
optimization process. All parameters with the same multiplier must have the same
dimensions.
"""
function add_optimization_multiplier!(dprm::DictParameters, targets...;
        initial = missing,
        lumping = missing,
        name = missing,
        abs_min = -Inf,
        abs_max = Inf
    )
    length(targets) > 0 || error("At least one target parameter must be provided for multiplier.")
    targets = map(t -> convert_key(t, dprm.parameters), targets)
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
    lumping = validate_and_normalize_lumping(lumping, initial, name)
    dprm.multipliers[name] = OptimizationMultiplier(abs_min, abs_max, collect(targets), lumping, initial)
    return dprm
end
