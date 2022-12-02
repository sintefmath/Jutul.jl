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

function simulator_storage(model; state0 = nothing,
                                parameters = setup_parameters(model),
                                copy_state = true,
                                mode = :forward,
                                specialize = false,
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
    return specialize_simulator_storage(storage, model, specialize)
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
    simulate!(sim::JutulSimulator, timesteps::AbstractVector; forces = nothing,
                                                                   config = nothing,
                                                                   initialize = true,
                                                                   restart = nothing,
                                                                   state0 = nothing,
                                                                   parameters = nothing,
                                                                   kwarg...)

Non-allocating (or perhaps less allocating) version of [`simulate!`](@ref).

# Arguments
- `initialize=true`: Perform internal updates as if this is the first time 

See also [`simulate`](@ref) for additional supported input arguments.
"""
function simulate!(sim::JutulSimulator, timesteps::AbstractVector; forces = setup_forces(sim.model),
                                                                   config = nothing,
                                                                   initialize = true,
                                                                   restart = nothing,
                                                                   state0 = nothing,
                                                                   parameters = nothing,
                                                                   kwarg...)
    if isnothing(config)
        config = simulator_config(sim; kwarg...)
    else
        # Reset recorder just in case since we are starting a new simulation
        config[:ProgressRecorder] = ProgressRecorder()
        for (k, v) in kwarg
            if !haskey(config, k)
                @warn "Keyword argument $k not found in initial config."
            end
            config[k] = v
        end
    end
    states, reports, first_step, dt = initial_setup!(sim, config, timesteps, restart = restart, state0 = state0, parameters = parameters)
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
    retrieve_output!(states, config, no_steps)
    return (states, reports)
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
            dt_old = dt
            dt = cut_timestep(sim, config, dt, dT, reports, step_index = step_no, cut_count = cut_count)
            if info_level > 0
                if isnan(dt)
                    inner_msg = " Aborting."
                    c = :red
                else
                    inner_msg = " Reducing mini-step."
                    c = :yellow
                end
                jutul_message("Convergence", "Time-step $step_no with mini-step length $(get_tstr(dt_old)) failed to converge.$inner_msg", color = c)
            end
            if isnan(dt)
                # Timestep too small, cut too many times, ...
                if info_level == 0 || info_level > 1
                    @error "Unable to reduce time-step to smaller value. Aborting simulation."
                end
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
        nextstep_local!(rec, dt, ok)
    end
    return (done, ministep_reports, dt)
end

function perform_step!(simulator::JutulSimulator, dt, forces, config; vararg...)
    perform_step!(simulator.storage, simulator.model, dt, forces, config; vararg...)
end

function perform_step!(storage, model, dt, forces, config; iteration = NaN, relaxation = 1, update_secondary = nothing, solve = true)
    if isnothing(update_secondary)
        update_secondary = iteration > 1 || config[:always_update_secondary]
    end
    e, converged = nothing, false

    report = OrderedDict()
    # Update the properties and equations
    t_asm = @elapsed begin
        time = config[:ProgressRecorder].recorder.time + dt
        update_state_dependents!(storage, model, dt, forces, time = time, update_secondary = update_secondary)
    end
    report[:assembly_time] = t_asm
    # Update the linearized system
    report[:linear_system_time] = @elapsed begin
        @timeit "linear system" update_linearized_system!(storage, model)
    end
    t_conv = @elapsed begin
        if iteration == config[:max_nonlinear_iterations]
            tf = config[:tol_factor_final_iteration]
        else
            tf = 1
        end
        @timeit "convergence" converged, e, errors = check_convergence(storage, model, config, iteration = iteration, dt = dt, tol_factor = tf, extra_out = true)
        il = config[:info_level]
        if il > 1
            get_convergence_table(errors, il, iteration, config)
        end
        converged = converged && iteration > config[:min_nonlinear_iterations]
        report[:converged] = converged
        report[:errors] = errors
    end
    report[:convergence_time] = t_conv

    if !converged && solve
        lsolve = config[:linear_solver]
        check = config[:safe_mode]
        rec = config[:ProgressRecorder]
        t_solve, t_update, n_iter, rep_lsolve, rep_update = solve_and_update!(storage, model, dt, linear_solver = lsolve, check = check, recorder = rec, relaxation = relaxation)
        report[:update] = rep_update
        report[:linear_solver] = rep_lsolve
        report[:linear_iterations] = n_iter
        report[:linear_solve_time] = t_solve
        report[:update_time] = t_update
    end
    return (e, converged, report)
end

function solve_ministep(sim, dt, forces, max_iter, cfg; skip_finalize = false, relaxation = 1.0)
    done = false
    rec = cfg[:ProgressRecorder]
    report = OrderedDict()
    report[:dt] = dt
    step_reports = []
    cur_time = current_time(rec)
    update_before_step!(sim, dt, forces, time = cur_time)
    for it = 1:(max_iter+1)
        next_iteration!(rec)
        do_solve = it <= max_iter
        e, done, r = perform_step!(sim, dt, forces, cfg, iteration = it, relaxation = relaxation, solve = do_solve)
        push!(step_reports, r)
        if done
            break
        end
        w0 = relaxation
        relaxation = select_nonlinear_relaxation(sim, cfg[:relaxation], step_reports, relaxation)
        if cfg[:info_level] > 1 && relaxation != w0
            jutul_message("Relaxation", "Changed from $w0 to $relaxation at iteration $it.", color = :yellow)
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
            @warn reason
            break
        end
        report[:failure] = failure
    end
    report[:steps] = step_reports
    report[:success] = done
    t_finalize = 0.0
    if !skip_finalize
        if done
            t_finalize = @elapsed update_after_step!(sim, dt, forces; time = cur_time + dt)
        else
            reset_state_to_previous_state!(sim)
        end
    end
    report[:finalize_time] = t_finalize

    return (done, report)
end

function initialize_before_first_timestep!(sim, first_dT; kwarg...)
    @timeit "solve" begin
        @timeit "secondary variables" update_secondary_variables!(sim.storage, sim.model)
    end
end

function initial_setup!(sim, config, timesteps; restart = nothing, parameters = nothing, state0 = nothing)
    # Timing stuff
    set_global_timer!(config[:extra_timing])
    # Set up storage
    reports = []
    states = Vector{Dict{Symbol, Any}}()
    pth = config[:output_path]
    initialize_io(pth)
    has_restart = !(isnothing(restart) || restart == 0 || restart == false)
    if has_restart
        state0, dt, first_step = deserialize_restart(pth, restart, states, reports, config)
        msg = "Restarting from step $first_step."
        state0_has_changed = true
    else
        state0_has_changed = !isnothing(state0)
        msg = "Starting from first step."
        first_step = 1
        dt = timesteps[first_step]
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
        recompute_state0_secondary = true
    end
    if state0_has_changed
        # state0 does not match sim, update it.
        # First, reset previous state
        reset_previous_state!(sim, state0)
        # Update current variables
        reset_variables!(sim, state0)
    end
    if recompute_state0_secondary
        update_secondary_variables!(sim.storage, sim.model, true)
    end
    return (states, reports, first_step, dt)
end

function deserialize_restart(pth, restart, states, reports, config)
    @assert !isnothing(pth) "output_path must be specified if restarts are enabled"
    if isa(restart, Bool)
        restart = maximum(valid_restart_indices(pth)) + 1
    end
    first_step = restart
    prev_step = restart - 1;
    state0, report0 = read_restart(pth, prev_step)
    read_results(pth, read_reports = true, read_states = config[:output_states], states = states, reports = reports, range = 1:prev_step);
    dt = report0[:ministeps][end][:dt]
    return (state0, dt, first_step)
end

reset_variables!(sim, vars; kwarg...) = reset_variables!(sim.storage, sim.model, vars; kwarg...)
reset_state_to_previous_state!(sim) = reset_state_to_previous_state!(sim.storage, sim.model)
reset_previous_state!(sim, state0) = reset_previous_state!(sim.storage, sim.model, state0)


function update_before_step!(sim, dt, forces; kwarg...)
    update_before_step!(sim.storage, sim.model, dt, forces; kwarg...)
end

function update_after_step!(sim, dt, forces; kwarg...)
    update_after_step!(sim.storage, sim.model, dt, forces; kwarg...)
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
        error("Number of forces must match the number of timesteps ($nt timesteps, $nf forces)")
    end
end
