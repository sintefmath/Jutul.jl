abstract type JutulSimulator end
struct Simulator <: JutulSimulator
    model::JutulModel
    storage::JutulStorage
end

"""
    Simulator(model; <kwarg>)

Set up a simulator object for a `model` that can be used by [`simulate!`](@ref).
To avoid manually instantiating the simulator, the non-mutating
[`simulate`](@ref) interface can be used instead.
"""
function Simulator(model; extra_timing = false, kwarg...)
    set_global_timer!(extra_timing)
    storage = simulator_storage(model; kwarg...)
    print_global_timer(extra_timing)
    Simulator(model, storage)
end

function Simulator(case::JutulCase; kwarg...)
    Simulator(case.model; state0 = deepcopy(case.state0), parameters = deepcopy(case.parameters), kwarg...)
end

struct SimResult
    states::AbstractVector
    reports::AbstractVector
    start_timestamp::DateTime
    end_timestamp::DateTime
    function SimResult(states, reports, start_time)
        nr = length(reports)
        ns = length(states)
        @assert ns == nr || ns == nr-1 || ns == 0 "Recieved $ns or $ns - 1 states different from $nr reports"
        return new(states, reports, start_time, now())
    end
end

mutable struct SolveRecorder
    step       # Step index in context
    iterations # Total iterations in context
    failed     # Failed iterations
    time       # Time - last converged. Current implicit level at time + dt
    iteration  # Current iteration (if applicable)
    dt         # Current timestep
    function SolveRecorder()
        new(0, 0, 0, 0.0, 0, NaN)
    end
end

mutable struct ProgressRecorder
    recorder
    subrecorder
    function ProgressRecorder()
        new(SolveRecorder(), SolveRecorder())
    end
end

export JutulConfig
mutable struct JutulOption
    default_value
    short_description::String
    long_description::Union{String, Missing}
    valid_types
    valid_values
end

struct JutulConfig <: AbstractDict{Symbol, Any}
    name::Union{Nothing, Symbol}
    values::OrderedDict{Symbol, Any}
    options::OrderedDict{Symbol, JutulOption}
end

"""
JutulConfig(name = nothing)

A configuration object that acts like a `Dict{Symbol,Any}` but contains
additional data to limit the valid keys and values to those added by [`add_option!`](@ref)
"""
function JutulConfig(name = nothing)
    if !isnothing(name)
        name = Symbol(name)
    end
    data = OrderedDict{Symbol, Any}()
    options = OrderedDict{Symbol, JutulOption}()
    return JutulConfig(name, data, options)
end
