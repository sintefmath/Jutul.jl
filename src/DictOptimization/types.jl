KEYTYPE = Union{String, Symbol}
LIMIT_TYPE = Union{Array{Float64, <:Any}, Float64}

Base.@kwdef mutable struct KeyLimits
    rel_min::LIMIT_TYPE = -Inf
    rel_max::LIMIT_TYPE = Inf
    abs_min::LIMIT_TYPE = -Inf
    abs_max::LIMIT_TYPE = Inf
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
