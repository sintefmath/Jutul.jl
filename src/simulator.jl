export newton_step, simulate
export Simulator, TervSimulator
using Printf
using Dates


abstract type TervSimulator end
struct Simulator <: TervSimulator
    model::TervModel
    storage::NamedTuple
end

function Simulator(model; state0 = setup_state(model), parameters = setup_parameters(model))
    storage = allocate_storage(model)
    storage["parameters"] = parameters
    storage["state0"] = state0
    storage["state"] = convert_state_ad(model, state0)
    # We convert the mutable storage (currently Dict) to immutable (NamedTuple)
    # This allows for much faster lookup in the simulation itself.
    storage = convert_to_immutable_storage(storage)
    initialize_storage!(storage, model)
    Simulator(model, storage)
end

function newton_step(simulator::TervSimulator; vararg...)
    newton_step(simulator.model, simulator.storage; vararg...)
end


function newton_step(model, storage; dt = nothing, linsolve = nothing, sources = nothing, iteration = nan)
    # Update the equations themselves - the AD bit
    if isa(model.context, SingleCUDAContext)
        # No idea why this is needed, but it is.
        CUDA.synchronize()
    end
    t_asm = @elapsed begin 
        update_equations!(model, storage, dt = dt, sources = sources)
    end
    @debug "Assembled equations in $t_asm seconds."
    # Update the linearized system
    t_lsys = @elapsed begin
        update_linearized_system!(model, storage)
    end
    @debug "Updated linear system in $t_lsys seconds."

    lsys = storage.LinearizedSystem
    eqs = storage.Equations
    tol = 1e-3

    converged = true
    e = 0
    for key in keys(eqs)
        errors, tscale = convergence_criterion(model, storage, eqs[key], lsys, dt = dt)
        for (index, e) in enumerate(errors)
            s = @sprintf("It %d: |R_%d| = %e\n", iteration, index, e)
            @debug s
        end
        converged = converged && all(errors .< tol*tscale)
        e = maximum([e, maximum(errors)])
    end
    if converged
        do_solve = iteration == 1
        @debug "Step converged."
    else
        do_solve = true
    end
    if do_solve
        t_solve = @elapsed solve!(lsys, linsolve)
        @debug "Solved linear system in $t_solve seconds."
        t_upd = @elapsed update_state!(model, storage)
        @debug "Updated state $t_upd seconds."

    end
    return (e, tol)
end

function simulate(sim::TervSimulator, timesteps::AbstractVector; maxIterations = 10, outputStates = true, sources = nothing, linsolve = nothing)
    states = []
    no_steps = length(timesteps)
    @info "Starting simulation"
    for (step_no, dT) in enumerate(timesteps)
        t_str =  Dates.canonicalize(Dates.CompoundPeriod(Second(dT)))
        @info "Solving step $step_no/$no_steps of length $t_str."
        dt = dT
        done = false
        t_local = 0
        cut_count = 0
        while !done
            ok = solve_ministep(sim, dt, maxIterations, linsolve, sources)
            if ok
                t_local += dt
                if t_local >= dT
                    break
                end
            else
                @warn "Cutting time-step."
                @assert cut_count < 5
                dt = min(dt/2, dT - t_local)
                cut_count += 1
            end
        end
        if outputStates
            store_output!(states, sim)
        end
    end
    return states
    @info "Simulation complete."
end

function solve_ministep(sim, dt, maxIterations, linsolve, sources)
    done = false
    for it = 1:maxIterations
        e, tol = newton_step(sim, dt = dt, iteration = it, sources = sources, linsolve = linsolve)
        done = e < tol
        if done
            break
        end
        if e > 1e10 || isinf(e) || isnan(e)
            @assert false "Timestep $step_no diverged."
            break
        end
    end
    # @assert done "Timestep did not complete in $maxIterations iterations."
    if done
        t_finalize = @elapsed update_after_step!(sim)
        @debug "Finalized in $t_finalize seconds."
    end
    return done
end

function update_after_step!(sim)
    storage = sim.storage
    state = storage.state
    state0 = storage.state0
    for key in keys(state)
        @. state0[key] = value(state[key])
    end
end

function store_output!(states, sim)
    storage = sim.storage
    state = storage.state
    state_out = deepcopy(storage.state0)
    for key in keys(state)
        @. state_out[key] = value(state[key])
    end
    push!(states, state_out)
end

function update_state!(model, storage)
    lsys = storage.LinearizedSystem
    state = storage.state

    offset = 0
    primary = get_primary_variables(model)
    for p in primary
        n = number_of_degrees_of_freedom(model, p)
        rng = (1:n) .+ offset
        update_state!(state, p, model, view(lsys.dx, rng))
        offset += n
    end
end