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
    timing_out = config[:debug_level] > 1
    # Update the properties and equations
    t_asm = @elapsed begin 
        update_state_dependents!(storage, model, dt, forces)
    end
    if timing_out
        @debug "Assembled equations in $t_asm seconds."
    end
    # Update the linearized system
    t_lsys = @elapsed begin
        update_linearized_system!(storage, model)
    end
    if timing_out
        @debug "Updated linear system in $t_lsys seconds."
    end
    converged, e, errors = check_convergence(storage, model, iteration = iteration, dt = dt, extra_out = true)
    if config[:info_level] > 0
        @info "Convergence report, iteration $iteration:"
        get_convergence_table(errors)
    end

    if converged
        do_solve = iteration == 1
        @debug "Step converged."
    else
        do_solve = true
    end
    if do_solve
        lsolve = config[:linear_solver]
        check = config[:safe_mode]
        t_solve, t_update = solve_and_update!(storage, model::TervModel, dt, linear_solver = lsolve, check = check)
        if timing_out
            @debug "Solved linear system in $t_solve seconds."
            @debug "Updated state $t_update seconds."
        end
    end
    return (e, converged)
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
    states = []
    no_steps = length(timesteps)
    maxIterations = config[:max_nonlinear_iterations]
    linsolve = config[:linear_solver]
    @info "Starting simulation"
    for (step_no, dT) in enumerate(timesteps)
        t_str =  Dates.canonicalize(Dates.CompoundPeriod(Millisecond(ceil(1000*dT))))
        @info "Solving step $step_no/$no_steps of length $t_str."
        dt = dT
        done = false
        t_local = 0
        cut_count = 0
        while !done
            ok = solve_ministep(sim, dt, forces, maxIterations, linsolve, config)
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
        end
        if config[:output_states]
            store_output!(states, sim)
        end
    end
    @info "Simulation complete."
    return states
end

function solve_ministep(sim, dt, forces, maxIterations, linsolve, cfg)
    done = false
    update_before_step!(sim, dt, forces)
    for it = 1:maxIterations
        e, done = perform_step!(sim, dt, forces, cfg, iteration = it)
        if done
            break
        end
        if e > 1e10
            @warn "Simulator produced very large residuals: $e."
            break
        elseif !isfinite(e)
            @warn "Simulator produced non-finite residuals: $e."
            break
        end
    end

    if done
        t_finalize = @elapsed update_after_step!(sim, dt, forces)
        if cfg[:debug_level] > 1
            @debug "Finalized in $t_finalize seconds."
        end
    else
        reset_to_previous_state!(sim.storage, sim.model)
    end
    return done
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
