KEYTYPE = Union{String, Symbol}
LIMIT_TYPE = Union{Array{Float64, <:Any}, Float64}

abstract type DictOptimizationScaler end

Base.@kwdef struct BaseLogScaler <: DictOptimizationScaler
    base_max::Float64 = Inf
    epsilon::Float64 = 1e-12
end

Base.@kwdef mutable struct KeyLimits
    rel_min::LIMIT_TYPE = -Inf
    rel_max::LIMIT_TYPE = Inf
    abs_min::LIMIT_TYPE = -Inf
    abs_max::LIMIT_TYPE = Inf
    scaler::Union{DictOptimizationScaler, Symbol, Missing} = missing
    lumping::Union{Array{Int, <:Any}, Missing} = missing
end

Base.@kwdef mutable struct OptimizationMultiplier
    abs_min::LIMIT_TYPE = -Inf
    abs_max::LIMIT_TYPE = Inf
    targets::Vector{Vector{KEYTYPE}} = Vector{Vector{KEYTYPE}}()
    lumping::Union{Array{Int, <:Any}, Missing} = missing
    value::Array{Float64, <:Any}
end

struct LimitBounds
    "Lower bound - for specific variable"
    min::Float64
    "Upper bound - for specific variable"
    max::Float64
    "Upper bound - for group of variables the variable belongs to"
    min_group::Float64
    "Lower bound - for group of variables the variable belongs to"
    max_group::Float64
    function LimitBounds(min_val, max_val, min_group, max_group)
        return new(
            convert(Float64, min_val),
            convert(Float64, max_val),
            convert(Float64, min_group),
            convert(Float64, max_group)
        )
    end
end

function LimitBounds(val, group)
    return LimitBounds(val.min, val.max, group.min, group.max)
end

mutable struct DictParameters
    "Initial value of parameters for optimization"
    parameters
    "Optimized parameters after running `optimize!`"
    parameters_optimized
    "A dictionary containing the optimization parameters and their targets. The keys are vectors of strings/symbols representing the parameter names, and the values are tuples with the initial value, lower bound, and upper bound."
    parameter_targets
    possible_targets
    multipliers
    multipliers_optimized
    strict::Bool
    verbose::Bool
    active_type
    setup_function
    history
    @doc"""
        DictParameters(parameters)
        DictParameters(parameters::AbstractDict, setup_function = missing;
                strict = true,
                verbose = true,
                active_type = Float64
            )

    Set up a `DictParameters` object for optimization. Optionally, the setup
    function that takes an instance with the same keys as `parameters` together
    with a `step_info` dictionary can be provided. The setup function should
    return a `JutulCase` set up from the parameters in the Dict.

    Optional keyword arguments:
    - `strict`: If true, the optimization will throw an error if any of the
      parameters are not set with at least one of the upper or lower bounds.
    - `verbose`: If true, the optimization will print information about the
      optimization process.
    - `active_type`: The type of the parameters that are considered active in
      the optimization. Defaults to `Float64`. This is used to determine which
      parameters are active and should be optimized. This means that all entries
      (and entries in nested dictionaries) of the `parameters` dictionary must
      be of this type or an array with this type as element type.
    """
function DictParameters(parameters::AbstractDict, setup_function = missing;
            strict = true,
            verbose = true,
            active_type = Float64
        )
        possible_targets = Jutul.AdjointsDI.setup_vectorize_nested(parameters; active_type = active_type)
        pkeys = possible_targets.names
        length(pkeys) > 0 || error("No targets found.")
        return new(
            deepcopy(parameters),
            missing,
            Jutul.OrderedDict{Vector{KEYTYPE}, KeyLimits}(),
            pkeys,
            Jutul.OrderedDict{KEYTYPE, OptimizationMultiplier}(),
            Jutul.OrderedDict{KEYTYPE, Any}(),
            strict,
            verbose,
            active_type,
            setup_function, missing
        )
    end
end

function Base.show(io::IO, t::MIME"text/plain", dopt::DictParameters)
    active_names = active_keys(dopt)
    inactive_names = inactive_keys(dopt)
    nmult = length(keys(dopt.multipliers))
    nact = length(active_names)
    ninact = length(inactive_names)
    println(io, "DictParameters with $(nact+ninact) parameters ($nact active), and $nmult multipliers:")
    print_optimization_overview(dopt; io = io, print_inactive = true)
end

struct DictParametersSampler
    parameters
    setup_function
    output_function
    objective
    simulator
    config
    setup
end

function DictParametersSampler(dopt::DictParameters, output_function = (case, result) -> result;
        simulator = missing,
        config = missing,
        objective = missing
    )
    parameters = deepcopy(dopt.parameters)
    setup = optimization_setup(dopt)
    if ismissing(simulator)
        case = dopt.setup_function(parameters, missing)
        simulator = Jutul.Simulator(case)
    end
    if ismissing(config)
        config = simulator_config(simulator, info_level = -1, output_substates = true)
    end
    return DictParametersSampler(parameters, dopt.setup_function, output_function, objective, simulator, config, setup)
end

struct JutulOptimizationProblem
    dict_parameters::DictParameters
    setup_function
    objective
    x0
    x_setup
    limits
    backend_arg::NamedTuple
    cache
    solution_history::Union{Bool, Symbol}
    print_parameters::Bool
    allow_errors::Bool
    gradient_scaling::Union{Bool, Float64}
    output_path::Union{Nothing, String}
    function JutulOptimizationProblem(dopt::DictParameters, objective, setup_fn = dopt.setup_function;
            backend_arg = missing,
            info_level = 0,
            deps::Symbol = :case,
            deps_ad::Symbol = :jutul,
            simulator = missing,
            config = missing,
            solution_history::Union{Bool, Symbol} = false,
            print_parameters::Bool = false,
            allow_errors::Bool = false,
            gradient_scaling = false,
            output_path = nothing
        )
        if !isnothing(output_path)
            mkpath(output_path)
        end
        if ismissing(backend_arg)
            backend_arg = NamedTuple()
        end
        backend_arg = setup_optimization_backend_kwarg(; deps = deps, deps_ad = deps_ad, backend_arg...)
        x0, x_setup, limits = optimization_setup(dopt)

        # Set up a cache for forward/backward sim
        adj_cache = setup_optimization_cache(dopt, simulator = simulator, config = config, info_level = info_level)

        if solution_history isa Symbol
            solution_history in (:full, :x) || error("solution_history must be :full or :x if a Symbol. Got $solution_history.")
        end
        return new(
            dopt,
            setup_fn,
            objective,
            x0,
            x_setup,
            limits,
            backend_arg,
            adj_cache,
            solution_history,
            print_parameters,
            allow_errors,
            gradient_scaling,
            output_path
        )
    end
end

function setup_optimization_backend_kwarg(;
        deps::Symbol = :case,
        deps_ad::Symbol = :jutul,
        sparsity_step_type = missing,
        kwarg...
    )
    deps_ad in (:di, :jutul) || error("deps_ad must be :di or :jutul. Got $deps_ad.")
    is_outer = deps == :parameters || deps == :parameters_and_state0
    if ismissing(sparsity_step_type)
        if is_outer
            sparsity_step_type = :unique_forces
        else
            sparsity_step_type = :all
        end
    end
    backend_arg = Dict{Symbol, Any}(
        :use_sparsity => true,
        :di_sparse => true,
        :single_step_sparsity => is_outer,
        :do_prep => true,
        :deps => deps,
        :deps_ad => deps_ad,
        :sparsity_step_type => sparsity_step_type
    )
    for (k, v) in pairs(kwarg)
        backend_arg[k] = v
    end
    return NamedTuple(backend_arg)
end

function evaluate(opt::JutulOptimizationProblem, x = opt.x0; gradient = true, extra_timing = false)
    dopt = opt.dict_parameters
    setup_fn = opt.setup_function
    objective = opt.objective
    x_setup = opt.x_setup
    adj_cache = opt.cache
    backend_arg = opt.backend_arg
    obj, dobj_dx = solve_and_differentiate_for_optimization(x, dopt, setup_fn, objective, x_setup, adj_cache;
        backend_arg = backend_arg,
        gradient = gradient,
        print_parameters = opt.print_parameters,
        allow_errors = opt.allow_errors,
        solution_history = opt.solution_history,
        gradient_scaling = opt.gradient_scaling,
        extra_timing = extra_timing,
        output_path = opt.output_path
    )
    return (obj, dobj_dx)
end

function (I::JutulOptimizationProblem)(x = I.x0; kwarg...)
    return evaluate(I, x; kwarg...)
end

function finite_difference_gradient_entry(I::JutulOptimizationProblem, x = I.x0; index = 1, eps = 1e-6)
    f0, _ = I(x; gradient = false)
    xd = copy(x)
    xd[index] += eps
    fd, _ = I(xd; gradient = false)
    return (fd - f0)/eps
end

function optimizer_devectorize(P::JutulOptimizationProblem, x)
    prm_out = deepcopy(P.dict_parameters.parameters)
    optimizer_devectorize!(prm_out, x, P.x_setup, multipliers = P.dict_parameters.multipliers_optimized)
    return prm_out
end
