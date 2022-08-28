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

function simulator_storage(model; state0 = nothing, parameters = setup_parameters(model), copy_state = true, mode = :forward, kwarg...)
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
    @timeit "state0" if isnothing(state0)
        state0 = setup_state(model)
    elseif copy_state
        # Take a deep copy to avoid side effects.
        state0 = deepcopy(state0)
    end
    @timeit "storage" storage = setup_storage(model, state0 = state0, parameters = parameters, state0_ad = state0_ad, state_ad = state_ad)
    # Initialize for first time usage
    @timeit "init_storage" initialize_storage!(storage, model; kwarg...)
    # We convert the mutable storage (currently Dict) to immutable (NamedTuple)
    # This allows for much faster lookup in the simulation itself.
    storage = convert_to_immutable_storage(storage)
    return storage
end

simulate(sim::JutulSimulator, timesteps::AbstractVector; kwarg...) = simulate!(sim, timesteps; kwarg...)

function simulate!(sim::JutulSimulator, timesteps::AbstractVector; forces = nothing, config = nothing, initialize = true, restart = nothing, kwarg...)
    if isnothing(config)
        config = simulator_config(sim; kwarg...)
    end
    states, reports, first_step, dt = initial_setup!(sim, config, timesteps, restart = restart)
    # Time-step info to keep around
    no_steps = length(timesteps)
    t_tot = sum(timesteps)
    # Config options
    max_its = config[:max_nonlinear_iterations]
    rec = config[:ProgressRecorder]
    info_level = config[:info_level]
    # Initialize loop
    p = start_simulation_message(info_level, timesteps)
    early_termination = false
    if initialize
        check_forces(sim, forces, timesteps)
        forces_step = forces_for_timestep(sim, forces, timesteps, first_step)
        initialize_before_first_timestep!(sim, dt, forces = forces_step, config = config)
    end
    for step_no = first_step:no_steps
        dT = timesteps[step_no]
        if early_termination
            break
        end
        forces_step = forces_for_timestep(sim, forces, timesteps, step_no)
        nextstep_global!(rec, dT)
        new_simulation_control_step_message(info_level, p, rec, step_no, no_steps, dT, t_tot)
        t_step = @elapsed step_done, rep, dt = solve_timestep!(sim, dT, forces_step, max_its, config; dt = dt, reports = reports, step_no = step_no, rec = rec)
        early_termination = !step_done
        subrep = OrderedDict()
        subrep[:ministeps] = rep
        subrep[:total_time] = t_step
        if step_done
            @timeit "output" store_output!(states, reports, step_no, sim, config, subrep)
        end
    end
    final_simulation_message(sim, p, reports, timesteps, config, early_termination)
    retrieve_output!(states, config)
    return (states, reports)
end


function solve_timestep!(sim, dT, forces, max_its, config; dt = dT, reports = nothing, step_no = NaN, 
                                                        info_level = config[:info_level],
                                                        rec = config[:ProgressRecorder], kwarg...)
    ministep_reports = []
    # Initialize time-stepping
    dt = pick_timestep(sim, config, dt, dT, reports, ministep_reports, step_index = step_no, new_step = true)
    done = false
    t_local = 0
    cut_count = 0
    ctr = 1
    nextstep_local!(rec, dt, false)
    while !done
        # Make sure that we hit the endpoint in case timestep selection is too optimistic.
        dt = min(dt, dT - t_local)
        # Attempt to solve current step
        @timeit "solve" ok, s = solve_ministep(sim, dt, forces, max_its, config; kwarg...)
        # We store the report even if it is a failure.
        push!(ministep_reports, s)
        if ok
            t_local += dt
            if t_local >= dT
                # Onto the next one
                done = true
                break
            else
                # Pick another for the next step...
                dt = pick_timestep(sim, config, dt, dT, reports, ministep_reports, step_index = step_no, new_step = false)
            end
        else
            if info_level > 0
                @info "🟡 Time-step $step_no of length $(get_tstr(dt)) failed to converge."
            end
            dt = cut_timestep(sim, config, dt, dT, reports, step_index = step_no, cut_count = cut_count)
            if isnan(dt)
                # Timestep too small, cut too many times, ...
                if info_level > -1
                    @error "Unable to reduce time-step to smaller value. Aborting simulation."
                end
                break
            else
                cut_count += 1
                if info_level > 0
                    @warn "Cutting timestep. Step $(100*t_local/dT) % complete.\nStep fraction reduced to $(get_tstr(dt)) ($(100*dt/dT)% of full step).\nThis is cut #$cut_count for step $step_no."
                end
            end
        end
        ctr += 1
        nextstep_local!(rec, dt, ok)
    end
    return (done, ministep_reports, dt)
end

function perform_step!(simulator::JutulSimulator, dt, forces, config; vararg...)
    perform_step!(simulator.storage, simulator.model, dt, forces, config; vararg...)
end

function perform_step!(storage, model, dt, forces, config; iteration = NaN, update_secondary = iteration > 1)
    do_solve, e, converged = true, nothing, false

    report = OrderedDict()
    timing_out = config[:debug_level] > 1
    # Update the properties and equations
    t_asm = @elapsed begin
        time = config[:ProgressRecorder].recorder.time + dt
        update_state_dependents!(storage, model, dt, forces, time = time, update_secondary = update_secondary)
    end
    if timing_out
        @debug "Assembled equations in $t_asm seconds."
    end
    report[:assembly_time] = t_asm
    # Update the linearized system
    report[:linear_system_time] = @elapsed begin
        @timeit "linear system" update_linearized_system!(storage, model)
    end
    if timing_out
        @debug "Updated linear system in $(report[:linear_system_time]) seconds."
    end
    t_conv = @elapsed begin
        @timeit "convergence" converged, e, errors = check_convergence(storage, model, config, iteration = iteration, dt = dt, extra_out = true)
        il = config[:info_level]
        if il > 1
            get_convergence_table(errors, il, iteration, config)
        end
        if converged
            if iteration <= config[:min_nonlinear_iterations]
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
        report[:converged] = converged
        report[:errors] = errors
    end
    report[:convergence_time] = t_conv

    if do_solve
        lsolve = config[:linear_solver]
        check = config[:safe_mode]
        rec = config[:ProgressRecorder]
        t_solve, t_update, n_iter, rep_lsolve = solve_and_update!(storage, model, dt, linear_solver = lsolve, check = check, recorder = rec)
        if timing_out
            @debug "Solved linear system in $t_solve seconds with $n_iter iterations."
            @debug "Updated state $t_update seconds."
        end
        report[:linear_solver] = rep_lsolve
        report[:linear_iterations] = n_iter
        report[:linear_solve_time] = t_solve
        report[:update_time] = t_update
    end
    return (e, converged, report)
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
    report[:success] = done
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

function initialize_before_first_timestep!(sim, first_dT; kwarg...)
    @timeit "solve" begin
        @timeit "secondary variables" update_secondary_variables!(sim.storage, sim.model)
    end
end

function initial_setup!(sim, config, timesteps; restart = nothing)
    # Timing stuff
    set_global_timer!(config[:extra_timing])
    # Set up storage
    reports = []
    states = Vector{Dict{Symbol, Any}}()
    pth = config[:output_path]
    do_print = config[:info_level] > 0
    initialize_io(pth)
    if isnothing(restart) || restart == 0 || restart == false
        if do_print
            @info "Starting from first step."
        end
        first_step = 1
        dt = timesteps[first_step]
    else
        @assert !isnothing(pth) "output_path must be specified if restarts are enabled"
        if isa(restart, Bool)
            restart = maximum(valid_restart_indices(pth)) + 1
        end
        first_step = restart
        prev_step = restart - 1;
        state0, report0 = read_restart(pth, prev_step)
        read_results(pth, read_reports = true, read_states = config[:output_states], states = states, reports = reports, range = 1:prev_step);
        reset_previous_state!(sim, state0)
        reset_to_previous_state!(sim)
        dt = report0[:ministeps][end][:dt]
        if do_print
            @info "Restarting from step $first_step."
        end
    end
    return (states, reports, first_step, dt)
end

reset_to_previous_state!(sim) = reset_to_previous_state!(sim.storage, sim.model)
reset_previous_state!(sim, state0) = reset_previous_state!(sim.storage, sim.model, state0)


function update_before_step!(sim, dt, forces)
    update_before_step!(sim.storage, sim.model, dt, forces)
end

function update_after_step!(sim, dt, forces)
    update_after_step!(sim.storage, sim.model, dt, forces)
end

# Forces - one for the entire sim
check_forces(sim, forces, timesteps) = nothing
forces_for_timestep(sim, f, timesteps, step_index) = f
# Forces as a vector - one per timestep
forces_for_timestep(sim, f::T, timesteps, step_index) where T<:AbstractArray = f[step_index]
function check_forces(sim, f::T, timesteps) where T<:AbstractArray
    nf = length(f)
    nt = length(timesteps)
    if nf != nt
        error("Number of forces must match the number of timesteps ($nt timsteps, $nf forces)")
    end
end
