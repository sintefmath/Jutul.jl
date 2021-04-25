export newton_step, simulate
export Simulator, TervSimulator
using Printf
using Dates


abstract type TervSimulator end
struct Simulator <: TervSimulator
    model::TervModel
    storage::Dict{String, Any}
end

function Simulator(model; state0 = setup_state(model), parameters = setup_parameters(model))
    storage = allocate_storage(model)
    storage["parameters"] = parameters
    storage["state0"] = state0
    storage["state"] = convert_state_ad(model, state0)
    Simulator(model, storage)
end

function newton_step(simulator::TervSimulator; vararg...)
    newton_step(simulator.model, simulator.storage; vararg...)
end


function newton_step(model, storage; dt = nothing, linsolve = nothing, sources = nothing, iteration = nan)

    # Update the equations themselves - the AD bit
    t_asm = @elapsed begin 
        update_equations!(model, storage, dt = dt, sources = sources)
    end
    @debug "Assembled equations in $t_asm seconds."
    # Update the linearized system
    t_lsys = @elapsed begin
        update_linearized_system!(model, storage)
    end
    @debug "Updated linear system in $t_lsys seconds."

    lsys = storage["LinearizedSystem"]
    e = norm(lsys.r, Inf)
    tol = 1e-3
    if e < tol
        do_solve = iteration == 1
        cstr = " -> Converged."
    else
        do_solve = true
        cstr = ""
    end
    s = @sprintf("It %d: |R| = %e%s\n", iteration, e, cstr)
    @info s

    if do_solve
        solve!(lsys, linsolve)
        storage["state"]["Pressure"] += lsys.dx
    end
    return (e, tol)
end

function simulate(sim::TervSimulator, timesteps::AbstractVector; maxIterations = 10, outputStates = true, sources = nothing, linsolve = nothing)
    states = []
    no_steps = length(timesteps)
    @info "Starting simulation"
    for (step_no, dt) in enumerate(timesteps)
        t_str =  Dates.canonicalize(Dates.CompoundPeriod(Second(dt)))
        @info "Solving step $step_no/$no_steps of length $t_str."
        done = false
        for it = 1:maxIterations
            e, tol = newton_step(sim, dt = dt, iteration = it, sources = sources, linsolve = linsolve)
            done = e < tol
            if done
                break
            end
            if e > 1e10 || isinf(e) || isnan(e)
                @assert "Timestep $step_no diverged."
                break
            end
        end
        @assert done "Timestep $step_no did not complete in $maxIterations iterations."
        if outputStates
            push!(states, value(sim.storage["state"]))
        end
    end
    return states
    @info "Simulation complete."
end
