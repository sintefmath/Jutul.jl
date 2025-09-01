iteration(r) = r.recorder.iteration
subiteration(r) = r.subrecorder.iteration
step(r) = r.recorder.step
substep(r) = r.subrecorder.step

# Progress recorder stuff
function recorder_start_step!(rec::ProgressRecorder, dt, level::Symbol)
    level == :local || level == :global || error("level must be :local or :global")
    if level == :local
        rec.subrecorder.dt = dt
    else # global
        rec.recorder.dt = dt
    end
end

function recorder_log_step!(rec::ProgressRecorder, success, level::Symbol)
    level == :local || level == :global || error("level must be :local or :global")
    l = rec.subrecorder
    g = rec.recorder

    function update!(solver_rec, success)
        if success
            solver_rec.step += 1
            solver_rec.time += solver_rec.dt
        else
            solver_rec.failed += solver_rec.iteration
        end
        solver_rec.iterations += solver_rec.iteration
        solver_rec.iteration = 0
    end

    if level == :local
        update!(l, success)
    else # global
        g.iteration = l.iterations
        g.failed += l.failed
        update!(g, success)
        recorder_reset!(l)
    end
end

function recorder_current_time(rec::ProgressRecorder, level::Symbol)
    level == :local || level == :global || error("level must be :local or :global")
    if level == :local
        return rec.subrecorder.time
    else # global
        return rec.recorder.time + rec.subrecorder.time
    end
end

function recorder_increment_iteration!(rec::ProgressRecorder, report, level::Symbol)
    level == :local || level == :global || error("level must be :local or :global")
    if level == :local
        if haskey(report, :update_time)
            rec.subrecorder.iteration += 1
        end
    end
end

function recorder_reset!(r::SolveRecorder, dt = NaN; step = 1, iterations = 0, iteration = 0, time = 0.0)
    r.step = step
    r.iterations = iterations
    r.time = time
    r.iteration = iteration
    r.dt = dt
end

function recorder_reset!(target::SolveRecorder, source::SolveRecorder)
    target.step = source.step
    target.iterations = source.iterations
    target.time = source.time
    target.iteration = source.iteration
    target.dt = source.dt
end

function recorder_reset!(r::ProgressRecorder, dt = NaN; kwarg...)
    recorder_reset!(r.recorder, dt; kwarg...)
    recorder_reset!(r.subrecorder, 0.0)
end

function recorder_reset!(target::ProgressRecorder, source::ProgressRecorder)
    recorder_reset!(target.recorder, source.recorder)
    recorder_reset!(target.subrecorder, source.recorder)
end

@deprecate reset!(r::SolveRecorder, dt = NaN;
    step = 1,
    iterations = 0,
    iteration = 0,
    time = 0.0) recorder_reset!(r, dt; step, iterations, iteration, time)

@deprecate reset!(target::SolveRecorder, source::SolveRecorder) recorder_reset!(targe, source)
@deprecate reset!(r::ProgressRecorder, dt = NaN; kwarg...) recorder_reset!(r, dt; kwarg...)
@deprecate reset!(target::ProgressRecorder, source::ProgressRecorder) recorder_reset!(target, source)
