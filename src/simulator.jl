export simulate, perform_step!
export Simulator, TervSimulator
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

function perform_step!(simulator::TervSimulator, dt, forces, config; vararg...)
    perform_step!(simulator.storage, simulator.model, dt, forces, config; vararg...)
end

function perform_step!(storage, model, dt, forces, config; iteration = NaN)
    do_solve, e, converged = true, nothing, false

    report = OrderedDict()
    timing_out = config[:debug_level] > 1
    # Update the properties and equations
    t_asm = @elapsed begin 
        update_state_dependents!(storage, model, dt, forces)
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
        if config[:info_level] > 1
            @info "Convergence report, iteration $iteration:"
            get_convergence_table(errors)
        end
        if converged
            do_solve = iteration == 1
            @debug "Step converged."
        end
    end

    if do_solve
        lsolve = config[:linear_solver]
        check = config[:safe_mode]
        t_solve, t_update = solve_and_update!(storage, model::TervModel, dt, linear_solver = lsolve, check = check)
        if timing_out
            @debug "Solved linear system in $t_solve seconds."
            @debug "Updated state $t_update seconds."
        end
        report[:linear_solve_time] = t_solve
        report[:update_time] = t_update
    end
    return (e, converged, report)
end

function simulator_config(sim; kwarg...)
    cfg = Dict()
    cfg[:max_timestep_cuts] = 5
    cfg[:max_nonlinear_iterations] = 15
    cfg[:linear_solver] = nothing
    cfg[:output_states] = true
    # Extra checks on values etc
    cfg[:safe_mode] = true
    # Define debug level. If debugging is on, this determines the amount of output.
    cfg[:debug_level] = 1
    cfg[:info_level] = 1
    # Overwrite with varargin
    for key in keys(kwarg)
        cfg[key] = kwarg[key]
    end
    return cfg
end

function simulate(sim::TervSimulator, timesteps::AbstractVector; forces = nothing, config = nothing, kwarg...)
    if isnothing(config)
        config = simulator_config(sim; kwarg...)
    end
    reports = []
    states = []
    no_steps = length(timesteps)
    maxIterations = config[:max_nonlinear_iterations]
    linsolve = config[:linear_solver]
    @info "Starting simulation"
    for (step_no, dT) in enumerate(timesteps)
        subrep = OrderedDict()
        ministep_reports = []
        t_step = @elapsed begin
            t_str =  Dates.canonicalize(Dates.CompoundPeriod(Millisecond(ceil(1000*dT))))
            @info "Solving step $step_no/$no_steps of length $t_str."
            dt = dT
            done = false
            t_local = 0
            cut_count = 0
            ctr = 1
            while !done
                ok, s = solve_ministep(sim, dt, forces, maxIterations, linsolve, config)
                push!(ministep_reports, s)
                if ok
                    t_local += dt
                    if t_local >= dT
                        break
                    end
                else
                    max_cuts = config[:max_timestep_cuts]
                    if cut_count + 1 > max_cuts
                        @warn "Unable to converge time step $step_no/$no_steps. Aborting."
                        return states
                    end
                    cut_count += 1
                    dt = min(dt/2, dT - t_local)
                    @warn "Cutting time-step. Step $(100*t_local/dT) % complete.\nStep fraction reduced to $(100*dt/dT)% of full step.\nThis is cut $cut_count of $max_cuts allowed."
                end
                ctr += 1
            end
            subrep[:ministeps] = ministep_reports
            push!(reports, subrep)
            if config[:output_states]
                store_output!(states, sim)
            end
        end
        subrep[:total_time] = t_step
    end
    stats = report_stats(reports)
    info_level = config[:info_level]
    if info_level >= 0
        @info "Simulation complete. Completed $(stats.steps) time-steps in $(stats.time_sum.total) seconds with $(stats.newtons) iterations."
        if info_level > 0
            print_stats(stats)
        end
    end
    return (states, reports)
end

function solve_ministep(sim, dt, forces, maxIterations, linsolve, cfg)
    done = false
    report = OrderedDict()
    report[:dt] = dt
    step_reports = []
    update_before_step!(sim, dt, forces)
    for it = 1:maxIterations
        e, done, r = perform_step!(sim, dt, forces, cfg, iteration = it)
        push!(step_reports, r)
        if done
            break
        end
        too_large = e > 1e10
        non_finite = !isfinite(e)
        failure = non_finite || too_large
        if failure
            if too_large
                reason = "Simulator produced very large residuals: $e."
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
    if done
        t_finalize = @elapsed update_after_step!(sim, dt, forces)
        if cfg[:debug_level] > 1
            @debug "Finalized in $t_finalize seconds."
        end
        report[:finalize_time] = t_finalize
    else
        reset_to_previous_state!(sim.storage, sim.model)
    end
    return (done, report)
end

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
