abstract type JutulSimulator end
struct Simulator <: JutulSimulator
    model::JutulModel
    storage::JutulStorage
end

function Simulator(model; extra_timing = false, kwarg...)
    set_global_timer!(extra_timing)
    storage = simulator_storage(model; kwarg...)
    print_global_timer(extra_timing)
    Simulator(model, storage)
end

function Simulator(case::JutulCase; kwarg...)
    Simulator(case.model; state0 = deepcopy(case.state0), parameters = deepcopy(case.parameters), kwarg...)
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
