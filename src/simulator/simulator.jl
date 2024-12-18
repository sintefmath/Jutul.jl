export simulate, simulate!, perform_step!
export Simulator, JutulSimulator, ProgressRecorder
export simulator_config

include("types.jl")
include("config.jl")
include("io.jl")
include("print.jl")
include("recorder.jl")
include("timesteps.jl")
include("utils.jl")
include("optimization.jl")
include("relaxation.jl")
include("helper.jl")

function simulator_storage(model; state0 = nothing,
                                parameters = setup_parameters(model),
                                copy_state = true,
                                mode = :forward,
                                specialize = false,
                                prepare_step_handler = missing,
                                kwarg...)
    if mode == :forward
        state_ad = true
        state0_ad = false
    elseif mode == :reverse
        state_ad = false
        state0_ad = true
    else
        state_ad = true
        state0_ad = true
        @assert mode == :sensitivities
    end
    # We need to sort the secondary variables according to their dependency ordering before simulating.
    sort_secondary_variables!(model)
    @tic "state0" if isnothing(state0)
        state0 = setup_state(model)
    elseif copy_state
        # Take a deep copy to avoid side effects.
        state0 = deepcopy(state0)
    end
    @tic "storage" storage = setup_storage(model, state0 = state0, parameters = parameters, state0_ad = state0_ad, state_ad = state_ad)
    # Add internal book keeping of time steps
    storage[:recorder] = ProgressRecorder()
    # Initialize for first time usage
    @tic "init_storage" begin
        initialize_storage!(storage, model; kwarg...)
        if !ismissing(prepare_step_handler)
            pstep_storage = prepare_step_storage(prepare_step_handler, storage, model)
            storage[:prepare_step_handler] = (prepare_step_handler, pstep_storage)
        end
    end
    # We convert the mutable storage (currently Dict) to immutable (NamedTuple)
    # This allows for much faster lookup in the simulation itself.
    return specialize_simulator_storage(storage, model, specialize)
end

"""
    prepare_step_storage(storage, model, ::Missing)

Initialize storage for prepare_step_handler.
"""
function prepare_step_storage(storage, model, ::Missing)
    return missing
end

function specialize_simulator_storage(storage::JutulStorage, model_or_nothing, specialize)
    if specialize
        out = convert_to_immutable_storage(storage)
    else
        for (k, v) in data(storage)
            storage[k] = convert_to_immutable_storage(v)
        end
        out = storage
    end
    return out
end
"""
    simulate(state0, model, timesteps, parameters = setup_parameters(model))
    simulate(state0, model, timesteps, info_level = 3)
    simulate(state0, model, timesteps; <keyword arguments>)


Simulate a set of `timesteps` with `model` for the given initial `state0` and optionally specific parameters.
Additional keyword arguments are passed onto [`simulator_config`](@ref) and [`simulate!`](@ref). This interface is
primarily for convenience, as all storage for the simulator is allocated upon use and discared upon return. If you
want to perform multiple simulations with the same model it is advised to instead instantiate [`Simulator`](@ref) 
and combine it with [`simulate!`](@ref).

# Arguments
- `state0::Dict`: initial state, typically created using [`setup_state`](@ref) for the `model` in use.
- `model::JutulModel`: model that describes the discretized system to solve, for example [`SimulationModel`](@ref) or [`MultiModel`](@ref).
- `timesteps::AbstractVector`: Vector of desired report steps. The simulator will perform time integration until `sum(timesteps)`
   is reached, providing outputs at the end of each report step.
- `parameters=setup_parameters(model)`: Optional overrides the default parameters for the model.
- `forces=nothing`: Either `nothing` (for no forces), a single set of forces from `setup_forces(model)` or a `Vector` of such forces with equal length to `timesteps`.
- `restart=nothing`: If an integer is provided, the simulation will attempt to restart from that step. Requires that `output_path` is provided here or in the `config`.
- `config=simulator_config(model)`: Configuration `Dict` that holds many fine grained settings for output, linear solver, time-steps, outputs etc.

Additional arguments are passed onto [`simulator_config`](@ref).

See also [`simulate!`](@ref), [`Simulator`](@ref), [`SimulationModel`](@ref), [`simulator_config`](@ref).
"""
function simulate(state0, model::JutulModel, timesteps::AbstractVector; parameters = setup_parameters(model), kwarg...)
    sim = Simulator(model, state0 = state0, parameters = parameters)
    return simulate!(sim, timesteps; kwarg...)
end

function simulate(case::JutulCase; kwarg...)
    sim = Simulator(case)
    return simulate!(sim, case.dt; forces = case.forces, kwarg...)
end

"""
    simulate(state0, sim::JutulSimulator, timesteps::AbstractVector; parameters = nothing, kwarg...)

Simulate a set of `timesteps` with `simulator` for the given initial `state0` and optionally specific parameters.
"""
function simulate(state0, sim::JutulSimulator, timesteps::AbstractVector; parameters = nothing, kwarg...)
    return simulate!(sim, timesteps; state0 = state0, parameters = parameters, kwarg...)
end

simulate(sim::JutulSimulator, timesteps::AbstractVector; kwarg...) = simulate!(sim, timesteps; kwarg...)

"""
    simulate!(sim::JutulSimulator, timesteps::AbstractVector;
        forces = nothing,
        config = nothing,
        initialize = true,
        restart = nothing,
        state0 = nothing,
        parameters = nothing,
        kwarg...
    )

Non-allocating (or perhaps less allocating) version of [`simulate!`](@ref).

# Arguments
- `initialize=true`: Perform internal updates as if this is the first time 

See also [`simulate`](@ref) for additional supported input arguments.
"""
function simulate!(sim::JutulSimulator, timesteps::AbstractVector;
        forces = setup_forces(sim.model),
        config = nothing,
        initialize = true,
        restart = nothing,
        state0 = nothing,
        parameters = nothing,
        forces_per_step = isa(forces, Vector),
        start_date = nothing,
        kwarg...
    )
    rec = progress_recorder(sim)
    # Reset recorder just in case since we are starting a new simulation
    reset!(rec)
    forces, forces_per_step = preprocess_forces(sim, forces)
    start_timestamp = now()
    if isnothing(config)
        config = simulator_config(sim; kwarg...)
    else
        for (k, v) in kwarg
            if !haskey(config, k)
                @warn "Keyword argument $k not found in initial config."
            end
            config[k] = v
        end
    end
    # Time-step info to keep around
    no_steps = length(timesteps)
    t_tot = sum(timesteps)
    states, reports, first_step, dt = initial_setup!(
        sim,
        config,
        timesteps,
        restart = restart,
        state0 = state0,
        parameters = parameters,
        nsteps = no_steps
    )
    # Config options
    max_its = config[:max_nonlinear_iterations]
    info_level = config[:info_level]
    # Initialize loop
    p = start_simulation_message(info_level, timesteps, config)
    early_termination = false
    stopnow = false
    if initialize && first_step <= no_steps
        check_forces(sim, forces, timesteps, per_step = forces_per_step)
        forces_step = forces_for_timestep(sim, forces, timesteps, first_step, per_step = forces_per_step)
        initialize_before_first_timestep!(sim, dt, forces = forces_step, config = config)
    end
    n_solved = no_steps
    t_elapsed = 0.0
    for step_no = first_step:no_steps
        dT = timesteps[step_no]
        forces_step = forces_for_timestep(sim, forces, timesteps, step_no, per_step = forces_per_step)
        nextstep_global!(rec, dT)
        new_simulation_control_step_message(info_level, p, rec, t_elapsed, step_no, no_steps, dT, t_tot, start_date)
        if config[:output_substates]
            substates = []
        else
            substates = missing
        end
        t_step = @elapsed step_done, rep, dt = solve_timestep!(sim, dT, forces_step, max_its, config;
            dt = dt,
            reports = reports,
            step_no = step_no,
            rec = rec,
            substates = substates
        )
        early_termination = !step_done
        subrep = JUTUL_OUTPUT_TYPE()
        subrep[:ministeps] = rep
        subrep[:total_time] = t_step
        if step_done
            lastrep = rep[end]
            if get(lastrep, :stopnow, false)
                # Something inside the solver told us to stop.
                subrep[:output_time] = 0.0
                push!(reports, subrep)
                stopnow = true
            else
                @tic "output" store_output!(states, reports, step_no, sim, config, subrep, substates = substates)
            end
        else
            subrep[:output_time] = 0.0
            push!(reports, subrep)
        end
        t_elapsed += t_step + subrep[:output_time]

        if early_termination
            n_solved = step_no-1
            break
        end

        if stopnow
            break
        end
    end
    states, reports = retrieve_output!(sim, states, reports, config, n_solved)
    final_simulation_message(sim, p, rec, t_elapsed, reports, timesteps, config, start_date, early_termination)
    return SimResult(states, reports, start_timestamp)
end


"""
    solve_timestep!(sim, dT, forces, max_its, config; <keyword arguments>)

Internal function for solving a single time-step with fixed driving forces.

# Arguments
- `sim`: `Simulator` instance.
- `dT`: time-step to be solved
- `forces`: Driving forces for the time-step
- `max_its`: Maximum number of steps/Newton iterations.
- `config`: Configuration for solver (typically output from `simulator_config`).

Note: This function is exported for fine-grained simulation workflows. The general [`simulate`](@ref) interface is
both easier to use and performs additional validation.
"""
function solve_timestep!(sim, dT, forces, max_its, config;
        dt = dT,
        reports = nothing,
        step_no = NaN,
        info_level = config[:info_level],
        rec = progress_recorder(sim),
        substates = missing,
        kwarg...
    )
    ministep_reports = []
    # Initialize time-stepping
    dt = pick_timestep(sim, config, dt, dT, forces, reports, ministep_reports, step_index = step_no, new_step = true)
    done = false
    t_local = 0
    cut_count = 0
    ctr = 1
    while !done
        # Make sure that we hit the endpoint in case timestep selection is too optimistic.
        dt = min(dt, dT - t_local)
        # Attempt to solve current step
        @tic "solve" ok, s = solve_ministep(sim, dt, forces, max_its, config; kwarg...)
        # We store the report even if it is a failure.
        push!(ministep_reports, s)
        nextstep_local!(rec, dt, ok)
        n_so_far = length(ministep_reports)
        if ok
            if 2 > info_level > 1
                jutul_message("Convergence", "Ministep #$n_so_far of $(get_tstr(dt, 1)) ($(round(100.0*dt/dT, digits=1))% of report step) converged.", color = :green)
            end
            t_local += dt
            if t_local >= dT
                # Onto the next one
                done = true
                break
            elseif get(s, :stopnow, false)
                done = true
                break
            else
                # Add to output of intermediate states.
                if !ismissing(substates)
                    push!(substates, get_output_state(sim.storage, sim.model))
                end
                # Pick another for the next step...
                dt = pick_timestep(sim, config, dt, dT, forces, reports, ministep_reports, step_index = step_no, new_step = false, remaining_time = dT - t_local)
            end
        else
            dt_old = dt
            dt = cut_timestep(sim, config, dt, dT, forces, reports, step_index = step_no, cut_count = cut_count)
            check_for_inner_exception(dt, s)
            if info_level > 0
                if isnan(dt)
                    inner_msg = " Aborting."
                    c = :red
                else
                    inner_msg = " Reducing mini-step."
                    c = :yellow
                end
                jutul_message("Convergence", "Report step $step_no, mini-step #$n_so_far ($(get_tstr(dt_old, 2))) failed to converge.$inner_msg", color = c)
            end
            if isnan(dt)
                break
            else
                cut_count += 1
                if info_level > 1
                    t_format = t -> @sprintf "%1.2f" 100*t/dT
                    @warn "Cutting mini-step. Step $(t_format(t_local)) % complete.\nΔt reduced to $(get_tstr(dt)) ($(t_format(dt))% of full time-step).\nThis is cut #$cut_count for time-step #$step_no."
                end
            end
        end
        ctr += 1
    end
    return (done, ministep_reports, dt)
end

function perform_step!(simulator::JutulSimulator, dt, forces, config; vararg...)
    perform_step!(simulator.storage, simulator.model, dt, forces, config; executor = simulator.executor, vararg...)
end

function perform_step!(storage, model, dt, forces, config;
        executor = default_executor(),
        iteration::Int = 0,
        relaxation::Float64 = 1.0,
        update_secondary = nothing,
        solve = true,
        report = setup_ministep_report(),
        prev_report = missing
    )
    if isnothing(update_secondary)
        update_secondary = iteration > 1 || config[:always_update_secondary]
    end
    # Update the properties and equations
    rec = storage.recorder
    time = rec.recorder.time + dt
    prep, prep_storage = get_prepare_step_handler(storage)
    # Apply a pre-step if it exists
    if ismissing(prep)
        t_secondary, t_eqs = update_state_dependents!(storage, model, dt, forces, time = time, update_secondary = update_secondary)
    else
        if update_secondary
            t_secondary = @elapsed update_secondary_variables!(storage, model)
        else
            t_secondary = 0.0
        end
        t_prep = @elapsed prep, forces = prepare_step!(prep_storage, prep,
            storage, model, dt, forces, config;
            executor = executor,
            iteration = iteration,
            relaxation = relaxation
        )
        report[:prepare_step] = prep
        report[:prepare_step_time] = t_prep
        _, t_eqs = update_state_dependents!(storage, model, dt, forces, time = time, update_secondary = false)
    end
    report[:secondary_time] = t_secondary
    report[:equations_time] = t_eqs
    # Update the linearized system
    t_lsys = @elapsed begin
        @tic "linear system" update_linearized_system!(storage, model, executor)
    end
    report[:linear_system_time] = t_lsys
    solved = false
    if config[:check_before_solve]
        t_conv = @elapsed e, converged = perform_step_check_convergence_impl!(report, prev_report, storage, model, config, dt, iteration)
        should_solve = !converged && solve
        if should_solve
            solved = true
            perform_step_solve_impl!(report, storage, model, config, dt, iteration, rec, relaxation, executor)
        end
    else
        perform_step_solve_impl!(report, storage, model, config, dt, iteration, rec, relaxation, executor)
        prev_report = report
        t_conv = @elapsed e, converged = perform_step_check_convergence_impl!(report, prev_report, storage, model, config, dt, iteration)
        solved = true
    end
    report[:solved] = solved
    extra_debug_output!(report, storage, model, config, iteration, dt)
    post_hook = config[:post_iteration_hook]
    if !ismissing(post_hook)
        converged = post_hook(converged, report, storage, model, dt, forces, config, iteration)
    end
    return (e, converged, report)
end

function perform_step_check_convergence_impl!(report, prev_report, storage, model, config, dt, iteration)
    converged = false
    e = NaN
    t_conv = @elapsed begin
        if iteration == config[:max_nonlinear_iterations]+1
            tf = config[:tol_factor_final_iteration]
        else
            tf = 1
        end
        if ismissing(prev_report)
            update_report = missing
        else
            update_report = get(prev_report, :update, missing)
        end
        @tic "convergence" converged, e, errors = check_convergence(
            storage,
            model,
            config,
            iteration = iteration,
            dt = dt,
            tol_factor = tf,
            extra_out = true,
            update_report = update_report)
        il = config[:info_level]
        if il > 1.5
            get_convergence_table(errors, il, iteration, config)
        end
        converged = converged && iteration > config[:min_nonlinear_iterations]
        report[:converged] = converged
        report[:errors] = errors
    end
    report[:convergence_time] = t_conv
    return (e, converged)
end

function perform_step_solve_impl!(report, storage, model, config, dt, iteration, rec, relaxation, executor)
    lsolve = config[:linear_solver]
    check = config[:safe_mode]
    try
        t_solve, t_update, n_iter, rep_lsolve, rep_update = solve_and_update!(
                storage, model, dt,
                linear_solver = lsolve,
                check = check,
                recorder = rec,
                relaxation = relaxation,
                executor = executor
            )
        report[:update] = rep_update
        report[:linear_solver] = rep_lsolve
        report[:linear_iterations] = n_iter
        report[:linear_solve_time] = t_solve
        report[:update_time] = t_update
    catch e
        if config[:failure_cuts_timestep] && !(e isa InterruptException)
            if config[:info_level] > 0
                @warn "Exception occured in solve: $e. Attempting to cut time-step since failure_cuts_timestep = true."
            end
            report[:failure_exception] = e
        else
            rethrow(e)
        end
    end
end

function perform_step_per_process_initial_update!(sim::JutulSimulator, dt, forces, config; kwarg...)
    return perform_step_per_process_initial_update!(sim.storage, sim.model, dt, forces, config; kwarg...)
end

function perform_step_per_process_initial_update!(storage, model, dt, forces, config;
        executor = default_executor(),
        update_secondary = nothing,
        iteration = 0,
        report = setup_ministep_report()
    )
    if isnothing(update_secondary)
        update_secondary = iteration > 1 || config[:always_update_secondary]
    end
    @tic "secondary variables" if update_secondary
        t_s = @elapsed update_secondary_variables!(storage, model)
        report[:secondary_time] = t_s
    end
    return report
end

function setup_ministep_report(; kwarg...)
    report = JUTUL_OUTPUT_TYPE()
    for k in [:secondary_time, :equations_time, :linear_system_time, :convergence_time]
        report[k] = 0.0
    end
    report[:solved] = true
    report[:converged] = true
    report[:errors] = missing
    for (k, v) in kwarg
        report[k] = v
    end
    return report
end

function solve_ministep(sim, dt, forces, max_iter, cfg;
        finalize = true,
        prepare = true,
        relaxation = 1.0,
        update_explicit = true
    )
    done = false
    rec = progress_recorder(sim)
    report = JUTUL_OUTPUT_TYPE()
    report[:dt] = dt
    step_reports = JUTUL_OUTPUT_TYPE[]
    cur_time = current_time(rec)
    t_prepare = @elapsed if prepare
        update_before_step!(sim, dt, forces, time = cur_time, recorder = rec, update_explicit = update_explicit)
    end
    step_report = missing
    for it = 1:(max_iter+1)
        do_solve = it <= max_iter
        e, done, step_report = perform_step!(sim, dt, forces, cfg,
                    iteration = it,
                    relaxation = relaxation,
                    solve = do_solve,
                    executor = simulator_executor(sim),
                    prev_report = step_report
        )
        push!(step_reports, step_report)
        if haskey(step_report, :failure_exception)
            inner_exception = step_report[:failure_exception]
            if cfg[:info_level] > 0
                @warn "Exception occurred in perform_step!" inner_exception
            end
            break
        end
        next_iteration!(rec, step_report)
        if done
            break
        end
        relaxation, early_stop = apply_nonlinear_strategy!(sim, dt, forces, it, max_iter, cfg, e, step_reports, relaxation)
        if early_stop
            break
        end
    end
    report[:steps] = step_reports
    report[:success] = done
    report[:prepare_time] = t_prepare

    # Call hook and potentially modify the result
    post_hook = cfg[:post_ministep_hook]
    if !ismissing(post_hook)
        done, report = post_hook(done, report, sim, dt, forces, max_iter, cfg)
    end

    # Finalize (either reset state or proceed to next step)
    report[:finalize_time] = @elapsed if finalize
        if done
            report[:post_update] = update_after_step!(sim, dt, forces; time = cur_time + dt)
        else
            reset_state_to_previous_state!(sim)
        end
    end

    return (done, report)
end

function initialize_before_first_timestep!(sim, first_dT; kwarg...)
    @tic "solve" begin
        @tic "secondary variables" update_secondary_variables!(sim.storage, sim.model)
    end
end

function initial_setup!(sim, config, timesteps; restart = nothing, parameters = nothing, state0 = nothing, nsteps = Inf)
    # Timing stuff
    set_global_timer!(config[:extra_timing])
    # Threading
    if Threads.nthreads() > 1
        PolyesterWeave.reset_workers!()
    end
    # Set up storage
    reports = []
    states = Vector{Dict{Symbol, Any}}()
    pth = config[:output_path]
    initialize_io(pth)
    has_restart = !(isnothing(restart) || restart === 0 || restart === 1 || restart == false)
    dt = timesteps[1]
    simulation_is_done = false
    if has_restart
        state0, dt, first_step = deserialize_restart(pth, state0, dt, restart, states, reports, config, nsteps)
        msg = "Restarting from step $first_step."
        simulation_is_done = first_step == nsteps+1
        state0_has_changed = first_step != 1 && !simulation_is_done
    else
        state0_has_changed = !isnothing(state0)
        msg = "Starting from first step."
        first_step = 1
    end
    if config[:info_level] > 1
        jutul_message("Jutul", msg, color = :light_green)
    end
    recompute_state0_secondary = state0_has_changed
    if !isnothing(parameters)
        # Parameters are aliased into state and might have AD there
        for k in [:state, :state0, :parameters]
            reset_variables!(sim, parameters, type = k)
        end
        recompute_state0_secondary = !simulation_is_done
    end
    if state0_has_changed && !simulation_is_done
        # state0 does not match sim, update it.
        # First, reset previous state
        reset_previous_state!(sim, state0)
        # Update current variables
        reset_variables!(sim, state0)
    end
    if recompute_state0_secondary
        s = get_simulator_storage(sim)
        m = get_simulator_model(sim)
        @tic "secondary variables (state0)" update_secondary_variables!(s, m, true)
    end
    return (states, reports, first_step, dt)
end

function deserialize_restart(pth, state0, dt, restart, states, reports, config, nsteps = Inf)
    @assert !isnothing(pth) "output_path must be specified if restarts are enabled"
    if isa(restart, Bool)
        restart_ix = valid_restart_indices(pth)
        if length(restart_ix) == 0
            restart = 1
        else
            restart = maximum(restart_ix) + 1
        end
        if nsteps isa Integer
            restart = min(restart, nsteps+1)
        end
    end
    if nsteps isa Integer
        @assert restart <= nsteps+1 "Restart was $restart but schedule contains $nsteps steps."
    end
    first_step = restart
    if first_step > 1
        prev_step = restart - 1;
        state0, report0 = read_restart(pth, prev_step)
        kept_reports = config[:in_memory_reports]
        rep_start = max(prev_step-kept_reports+1, 1)
        for i in 1:(rep_start-1)
            push!(reports, missing)
        end
        read_results(pth, read_reports = true, read_states = false, states = states, reports = reports, range = rep_start:prev_step);
        dt = report0[:ministeps][end][:dt]
    end
    return (state0, dt, first_step)
end

function reset_variables!(sim, vars; kwarg...)
    s = get_simulator_storage(sim)
    m = get_simulator_model(sim)
    reset_variables!(s, m, vars; kwarg...)
end

function reset_state_to_previous_state!(sim)
    s = get_simulator_storage(sim)
    m = get_simulator_model(sim)
    reset_state_to_previous_state!(s, m)
end

function reset_previous_state!(sim, state0)
    s = get_simulator_storage(sim)
    m = get_simulator_model(sim)
    reset_previous_state!(s, m, state0)
end

function update_before_step!(sim, dt, forces; kwarg...)
    s = get_simulator_storage(sim)
    m = get_simulator_model(sim)
    update_before_step!(s, m, dt, forces; kwarg...)
end

function update_after_step!(sim, dt, forces; kwarg...)
    s = get_simulator_storage(sim)
    m = get_simulator_model(sim)
    update_after_step!(s, m, dt, forces; kwarg...)
end

function preprocess_forces(sim, forces)
    return (forces = forces, forces_per_step = forces isa Vector)
end

# Forces - one for the entire sim
function check_forces(sim, forces, timesteps; per_step = false)
    nothing
end

function forces_for_timestep(sim, f::Union{AbstractDict, Nothing, NamedTuple}, timesteps, step_index; per_step = false)
    f
end

function forces_for_timestep(sim, f::Vector, timesteps, step_index; per_step = true)
    if per_step
        force = f[step_index]
    else
        force = f
    end
    return force
end

function check_forces(sim::Simulator, f::Vector, timesteps; per_step = true)
    nf = length(f)
    nt = length(timesteps)
    if nf != nt && per_step
        error("Number of forces must match the number of timesteps ($nt timesteps, $nf forces)")
    end
end

function apply_nonlinear_strategy!(sim, dt, forces, it, max_iter, cfg, e, step_reports, relaxation)
    report = step_reports[end]
    w0 = relaxation
    relaxation = select_nonlinear_relaxation(sim, cfg[:relaxation], step_reports, relaxation)
    if cfg[:info_level] > 2 && relaxation != w0
        jutul_message("Relaxation", "Changed from $w0 to $relaxation at iteration $it.", color = :green)
    end
    failure = false
    max_res = cfg[:max_residual]
    if !isfinite(e)
        reason = "Simulator produced non-finite residuals: $e."
        failure = true
    elseif e > max_res
        reason = "Simulator produced very large residuals: $e larger than :max_residual=$max_res."
        failure = true
    else
        reason = ""
    end
    if failure
        report[:failure_message] = reason
        if cfg[:info_level] > 0
            @warn reason
        end
    end
    report[:failure] = failure
    cut_crit = cfg[:cutting_criterion]
    relaxation, early_stop = cutting_criterion(cut_crit, sim, dt, forces, it, max_iter, cfg, e, step_reports, relaxation)
    do_stop = failure || early_stop
    return (relaxation, do_stop)
end

function cutting_criterion(::Nothing, sim, dt, forces, it, max_iter, cfg, e, step_reports, relaxation)
    return (relaxation, false)
end

"""
    extra_debug_output!(report, storage, model, config, iteration, dt)

Add extra debug output to report during a nonlinear iteration.
"""
function extra_debug_output!(report, storage, model, config, iteration, dt)
    level = config[:debug_level]
    if level > 0
        debug_report = Dict{Symbol, Any}()
        report[:debug] = debug_report
        for i = 1:level
            extra_debug_output!(debug_report, report, storage, model, config, iteration, dt, Val(i))
        end
    end

end

function extra_debug_output!(debug_report, report, storage, model, config, iteration, dt, level::Val)
    # Default: Do nothing
end

function extra_debug_output!(debug_report, report, storage, model::Union{SimulationModel, MultiModel}, config, iteration, dt, level::Val{10})
    if haskey(storage, :LinearizedSystem)
        lsys = storage.LinearizedSystem
        r = vector_residual(lsys)
        debug_report[:linearized_system_norm] = (L1 = norm(r, 1), L2 = norm(r, 2), LInf = norm(r, Inf))
    end
end

function check_for_inner_exception(dt, s)
    if isnan(dt) && length(s[:steps]) > 0
        last_step = s[:steps][end]
        if haskey(last_step, :failure_exception)
            @error "Exception caught in perform_step and cannot cut time-step any further."
            throw(last_step[:failure_exception])
        end
    end
end

function prepare_step!(storage, model, dt, forces, config, ::Missing;
        executor = DefaultExecutor(),
        iteration = 0,
        relaxation = 1.0
    )
    return (nothing, forces)
end
