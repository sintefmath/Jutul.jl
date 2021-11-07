export simulate, perform_step!
export Simulator, TervSimulator, ProgressRecorder
export simulator_config

abstract type TervSimulator end
struct Simulator <: TervSimulator
    model::TervModel
    storage::TervStorage
end

function Simulator(model; state0 = nothing, parameters = setup_parameters(model), copy_state = true, kwarg...)
    # We need to sort the secondary variables according to their dependency ordering before simulating.
    sort_secondary_variables!(model)
    if isnothing(state0)
        state0 = setup_state(model)
    elseif copy_state
        # Take a deep copy to avoid side effects.
        state0 = deepcopy(state0)
    end
    storage = setup_storage(model, state0 = state0, parameters = parameters)
    # Initialize for first time usage
    initialize_storage!(storage, model; kwarg...)
    # We convert the mutable storage (currently Dict) to immutable (NamedTuple)
    # This allows for much faster lookup in the simulation itself.
    storage = convert_to_immutable_storage(storage)
    Simulator(model, storage)
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

function Base.show(io::IO, t::MIME"text/plain", sim::Simulator) 
    println("Simulator:")
    for f in fieldnames(typeof(sim))
        p = getfield(sim, f)
        print("  $f:\n")
        if f == :storage
            for key in keys(sim.storage)
                ss = sim.storage[key]
                println("    $key")
            end
        else
            show(io, t, p)
        end
    end
end

function simulator_config(sim; kwarg...)
    cfg = Dict()
    simulator_config!(cfg, sim; kwarg...)
    return cfg
end

function simulator_config!(cfg, sim; kwarg...)
    cfg[:max_timestep_cuts] = 5
    cfg[:max_nonlinear_iterations] = 15
    cfg[:min_nonlinear_iterations] = 1
    cfg[:linear_solver] = nothing
    cfg[:output_states] = true
    # Extra checks on values etc
    cfg[:safe_mode] = true
    # Define debug level. If debugging is on, this determines the amount of output.
    cfg[:debug_level] = 1
    cfg[:info_level] = 1
    # Define a default progress ProgressRecorder
    cfg[:ProgressRecorder] = ProgressRecorder()
    cfg[:timestep_selectors] = [TimestepSelector()]
    cfg[:timestep_max_increase] = 10.0
    cfg[:timestep_max_decrease] = 0.1
    # Max residual before error is issued
    cfg[:max_residual] = 1e20

    overwrite_by_kwargs(cfg; kwarg...)
    return cfg
end

function simulate(sim::TervSimulator, timesteps::AbstractVector; forces = nothing, config = nothing, kwarg...)
    if isnothing(config)
        config = simulator_config(sim; kwarg...)
    end
    reports = []
    states = Vector{Dict{Symbol, Any}}()
    no_steps = length(timesteps)
    max_its = config[:max_nonlinear_iterations]
    rec = config[:ProgressRecorder]
    output = config[:info_level] >= 0
    if output
        @info "Starting simulation"
    end
    early_termination = false
    dt = timesteps[1]
    for (step_no, dT) in enumerate(timesteps)
        if early_termination
            break
        end
        nextstep_global!(rec, dT)
        subrep = OrderedDict()
        ministep_reports = []
        subrep[:ministeps] = ministep_reports
        # Insert so that the current reports are accessible
        push!(reports, subrep)
        t_step = @elapsed begin
            if output
                @info "Solving step $step_no/$no_steps of length $(get_tstr(dT))."
            end
            dt = pick_timestep(sim, config, dt, dT, reports, step_index = step_no, new_step = true)
            done = false
            t_local = 0
            cut_count = 0
            ctr = 1
            nextstep_local!(rec, dt, false)
            while !done
                # Make sure that we hit the endpoint in case timestep selection is too optimistic.
                dt = min(dt, dT - t_local)
                # Attempt to solve current step
                ok, s = solve_ministep(sim, dt, forces, max_its, config)
                # We store the report even if it is a failure.
                push!(ministep_reports, s)
                if ok
                    t_local += dt
                    if t_local >= dT
                        # Onto the next one
                        break
                    else
                        # Pick another for the next step...
                        dt = pick_timestep(sim, config, dt, dT, reports, step_index = step_no, new_step = false)
                    end
                else
                    dt = cut_timestep(sim, config, dt, dT, reports, step_index = step_no, cut_count = cut_count)
                    if isnan(dt)
                        # Timestep too small, cut too many times, ...
                        if output
                            @warn "Unable to converge time step $step_no/$no_steps. Aborting."
                        end
                        early_termination = true
                        break
                    elseif output
                        cut_count += 1
                        @warn "Cutting timestep. Step $(100*t_local/dT) % complete.\nStep fraction reduced to $(100*dt/dT)% of full step.\nThis is cut #$cut_count.."
                    end
                end
                ctr += 1
                nextstep_local!(rec, dt, ok)
            end
            if config[:output_states]
                store_output!(states, sim)
            end
        end
        subrep[:total_time] = t_step
    end
    final_simulation_message(sim, reports, timesteps, config[:info_level], early_termination)
    return (states, reports)
end

function perform_step!(simulator::TervSimulator, dt, forces, config; vararg...)
    perform_step!(simulator.storage, simulator.model, dt, forces, config; vararg...)
end

function perform_step!(storage, model, dt, forces, config; iteration = NaN)
    do_solve, e, converged = true, nothing, false

    report = OrderedDict()
    timing_out = config[:debug_level] > 1
    # Update the properties and equations
    t_asm = @elapsed begin
        time =  config[:ProgressRecorder].recorder.time + dt
        update_state_dependents!(storage, model, dt, forces; time = time)
    end
    if timing_out
        @debug "Assembled equations in $t_asm seconds."
    end
    report[:assembly_time] = t_asm
    # Update the linearized system
    report[:linear_system_time] = @elapsed begin
        update_linearized_system!(storage, model)
    end
    if timing_out
        @debug "Updated linear system in $(report[:linear_system_time]) seconds."
    end
    report[:convergence_time] = @elapsed begin
        converged, e, errors = check_convergence(storage, model, iteration = iteration, dt = dt, extra_out = true)
        il = config[:info_level]
        if il > 1
            @info "Convergence report, iteration $iteration:"
            get_convergence_table(errors, il)
        end
        if converged
            if iteration < config[:min_nonlinear_iterations]
                # Should always do at least 
                do_solve = true
                # Ensures secondary variables are updated, and correct error
                converged = false
            else
                do_solve = false
                @debug "Step converged."
            end
        else
            do_solve = true
        end
    end

    if do_solve
        lsolve = config[:linear_solver]
        check = config[:safe_mode]
        rec = config[:ProgressRecorder]
        t_solve, t_update = solve_and_update!(storage, model::TervModel, dt, linear_solver = lsolve, check = check, recorder = rec)
        if timing_out
            @debug "Solved linear system in $t_solve seconds."
            @debug "Updated state $t_update seconds."
        end
        report[:linear_solve_time] = t_solve
        report[:update_time] = t_update
    end
    return (e, converged, report)
end

function overwrite_by_kwargs(cfg; kwarg...)
    # Overwrite with varargin
    for key in keys(kwarg)
        if !haskey(cfg, key)
            @warn "Key $key is not found in default config. Misspelled?"
        end
        cfg[key] = kwarg[key]
    end
end

function final_simulation_message(simulator, reports, timesteps, info_level, aborted)
    if info_level >= 0 && length(reports) > 0
        stats = report_stats(reports)
        if aborted
            endstr = "Simulation aborted. Completed $(stats.steps-1) of $(length(timesteps)) timesteps"
        else
            endstr = "Simulation complete. Completed $(stats.steps) timesteps"
        end
        @info "$endstr in $(stats.time_sum.total) seconds with $(stats.newtons) iterations."
        if info_level > 0
            print_stats(stats)
        end
    end
end

function pick_timestep(sim, config, dt_prev, dT, reports; step_index = NaN, new_step = false)
    dt = dT
    for sel in config[:timestep_selectors]
        if new_step && step_index == 1
            candidate = pick_first_timestep(sel, sim, config, dT)
        else
            candidate = pick_next_timestep(sel, sim, config, dt, dT, reports, step_index, new_step)
        end
        # @info "Found:" get_tstr(candidate) get_tstr(dt_prev) get_tstr(dt) sel.increase
        dt = min(dt, candidate)
    end
    # The selectors might go crazy, so we have some safety bounds
    if isfinite(dt_prev)
        min_allowable = config[:timestep_max_decrease]*dt_prev
        max_allowable = config[:timestep_max_increase]*dt_prev
        dt = min(max(dt, min_allowable), max_allowable)
    end
    if config[:info_level] > 1
        @info "Selected new sub-timestep $(get_tstr(dt)) from previous $(get_tstr(dt_prev))."
    end
    return dt
end

function cut_timestep(sim, config, dt, dT, reports; step_index = NaN, cut_count = 0)
    for sel in config[:timestep_selectors]
        candidate = pick_cut_timestep(sel, sim, config, dt, dT, reports, cut_count)
        dt = min(dt, candidate)
    end
    return dt
end

function get_tstr(dT)
    Dates.canonicalize(Dates.CompoundPeriod(Millisecond(ceil(1000*dT))))
end

function solve_ministep(sim, dt, forces, max_iter, cfg; skip_finalize = false)
    done = false
    rec = cfg[:ProgressRecorder]
    report = OrderedDict()
    report[:dt] = dt
    step_reports = []
    update_before_step!(sim, dt, forces)
    for it = 1:max_iter
        next_iteration!(rec)
        e, done, r = perform_step!(sim, dt, forces, cfg, iteration = it)
        push!(step_reports, r)
        if done
            break
        end
        too_large = e > cfg[:max_residual]
        non_finite = !isfinite(e)
        failure = non_finite || too_large
        if failure
            if too_large
                reason = "Simulator produced very large residuals: $e larger than :max_residual $(cfg[:max_residual])."
            else
                reason = "Simulator produced non-finite residuals: $e."
            end
            report[:failure_message] = reason
            @warn reason
            break
        end
        report[:failure] = failure
    end
    report[:steps] = step_reports
    if skip_finalize
        report[:finalize_time] = 0.0
    elseif done
        t_finalize = @elapsed update_after_step!(sim, dt, forces)
        if cfg[:debug_level] > 1
            @debug "Finalized in $t_finalize seconds."
        end
        report[:finalize_time] = t_finalize
    else
        reset_to_previous_state!(sim)
    end
    return (done, report)
end

reset_to_previous_state!(sim) = reset_to_previous_state!(sim.storage, sim.model)

function update_before_step!(sim, dt, forces)
    update_before_step!(sim.storage, sim.model, dt, forces)
end

function update_after_step!(sim, dt, forces)
    update_after_step!(sim.storage, sim.model, dt, forces)
end

function store_output!(states, sim)
    state_out = get_output_state(sim.storage, sim.model)
    push!(states, state_out)
end

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

function next_iteration!(rec)
    rec.subrecorder.iteration += 1
end

iteration(r) = r.recorder.iteration
subiteration(r) = r.subrecorder.iteration
step(r) = r.recorder.step
substep(r) = r.subrecorder.step

# Solve recorder stuff
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

function reset!(r::SolveRecorder, dt = NaN)
    r.step = 1
    r.iterations = 0
    r.time = 0.0
    r.iteration = 0
    r.dt = dt
end


