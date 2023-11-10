iteration(r) = r.recorder.iteration
subiteration(r) = r.subrecorder.iteration
step(r) = r.recorder.step
substep(r) = r.subrecorder.step

# Progress recorder stuff
function nextstep_global!(r::ProgressRecorder, dT, prev_success = !isnan(r.recorder.dt))
    g = r.recorder
    l = r.subrecorder
    g.iteration = l.iterations
    # A bit dicey, just need to get this working
    g.failed += l.failed
    nextstep!(g, dT, prev_success)
    reset!(r.subrecorder)
end

function nextstep_local!(r::ProgressRecorder, dT, prev_success = !isnan(r.local_recorder.dt))
    nextstep!(r.subrecorder, dT, prev_success)
end

function next_iteration!(rec, report)
    if haskey(report, :update_time)
        rec.subrecorder.iteration += 1
    end
end

function nextstep!(l::SolveRecorder, dT, success)
    # Update time
    if success
        l.step += 1
        l.time += l.dt
    else
        l.failed += l.iteration
    end
    l.dt = dT
    # Update iterations
    l.iterations += l.iteration
    l.iteration = 0
end

function reset!(r::SolveRecorder, dt = NaN; step = 1, iterations = 0, iteration = 0, time = 0.0)
    r.step = step
    r.iterations = iterations
    r.time = time
    r.iteration = iteration
    r.dt = dt
end

function reset!(target::SolveRecorder, source::SolveRecorder)
    target.step = source.step
    target.iterations = source.iterations
    target.time = source.time
    target.iteration = source.iteration
    target.dt = source.dt
end

function reset!(r::ProgressRecorder, dt = NaN; kwarg...)
    reset!(r.recorder, dt; kwarg...)
    reset!(r.subrecorder, 0.0)
end

function reset!(target::ProgressRecorder, source::ProgressRecorder)
    reset!(target.recorder, source.recorder)
    reset!(target.subrecorder, source.recorder)
end

function current_time(r::ProgressRecorder)
    return r.recorder.time + r.subrecorder.time
end

