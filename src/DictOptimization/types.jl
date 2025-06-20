KEYTYPE = Union{String, Symbol}
LIMIT_TYPE = Union{Array{Float64, <:Any}, Float64}

Base.@kwdef mutable struct KeyLimits
    rel_min::LIMIT_TYPE = -Inf
    rel_max::LIMIT_TYPE = Inf
    abs_min::LIMIT_TYPE = -Inf
    abs_max::LIMIT_TYPE = Inf
    scaler::Union{Symbol, Missing} = missing
    lumping::Union{Array{Int, <:Any}, Missing} = missing
end

mutable struct DictParameters
    parameters
    ""
    parameters_optimized
    "A dictionary containing the optimization parameters and their targets. The keys are vectors of strings/symbols representing the parameter names, and the values are tuples with the initial value, lower bound, and upper bound."
    parameter_targets
    possible_targets
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
    println(io, "DictParameters with $(length(active_names)) active parameters and $(length(active_names)) inactive:")
    print_optimization_overview(dopt; io = io, print_inactive = true)
end
