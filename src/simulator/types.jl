abstract type JutulSimulator end

function set_default_tolerances(sim::JutulSimulator; kwarg...)
    set_default_tolerances(sim.model; kwarg...)
end

abstract type JutulBackend end

struct DefaultBackend <: JutulBackend end

abstract type JutulExecutor end
struct DefaultExecutor <: JutulExecutor end
struct Simulator{E, M, S} <: JutulSimulator
    executor::E
    model::M
    storage::S
end

"""
    Simulator(model; <kwarg>)

Set up a simulator object for a `model` that can be used by [`simulate!`](@ref).
To avoid manually instantiating the simulator, the non-mutating
[`simulate`](@ref) interface can be used instead.
"""
function Simulator(model; extra_timing = false, executor = default_executor(), kwarg...)
    model::JutulModel
    set_global_timer!(extra_timing)
    storage = simulator_storage(model; kwarg...)
    storage::JutulStorage
    print_global_timer(extra_timing)
    return Simulator(executor, model, storage)
end

default_executor() = DefaultExecutor()
simulator_executor(sim) = sim.executor

function Simulator(case::JutulCase; kwarg...)
    return Simulator(case.model; state0 = deepcopy(case.state0), parameters = deepcopy(case.parameters), kwarg...)
end

function get_simulator_model(sim)
    return sim.model
end

function get_simulator_storage(sim)
    return sim.storage
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
    step::Int       # Step index in context
    iterations::Int # Total iterations in context
    failed::Int     # Failed iterations
    time::Float64   # Time - last converged. Current implicit level at time + dt
    iteration::Int  # Current iteration (if applicable)
    dt::Float64     # Current timestep
    function SolveRecorder()
        new(0, 0, 0, 0.0, 0, NaN)
    end
end

struct ProgressRecorder
    recorder::SolveRecorder
    subrecorder::SolveRecorder
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
