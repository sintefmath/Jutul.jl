
function Jutul.simulator_config(sim::PArraySimulator; extra_timing = false, kwarg...)
    cfg = JutulConfig("Simulator config")
    v = sim.storage.verbose
    if v
        il = 1
    else
        il = -1
    end

    is_mpi_win = isa(sim, MPISimulator) && Sys.iswindows()
    extra_timing = extra_timing && v
    cfg = Jutul.simulator_config!(cfg, sim;
        kwarg...,
        info_level = il,
        ascii_terminal = is_mpi_win,
        extra_timing = extra_timing
    )
    simulators = sim.storage[:simulators]
    output_pth = cfg[:output_path]
    write_output = !isnothing(output_pth)
    configs = map(simulators) do sim
        sd = sim.executor.data
        rank = sim.executor.rank
        # If output is requested we immediately write the partition to a subdir.
        # This makes consolidation of distributed states easier afterwards.
        if write_output
            np = sd[:number_of_processes]
            if np > 1
                pth = rank_folder(output_pth, rank)
                if !isdir(pth)
                    @assert isa(pth, String)
                    @debug "Creating $pth for output."
                    mkpath(pth)
                end
                jldopen(joinpath(pth, "partition.jld2"), "w") do file
                    file["partition"] = sd[:partition]
                    file["main_partition_label"] = sd[:main_label]
                    file["n_self"] = sd[:n_self]
                    file["n_total"] = sd[:n_total]
                    file["rank"] = rank
                end
            else
                # One process in total - we can write as normal.
                pth = output_pth
            end
        else
            pth = nothing
        end
        subconfig = Jutul.simulator_config(sim,
            info_level = -1,
            extra_timing = extra_timing,
            output_path = pth
        )
        subconfig
    end
    add_option!(cfg, :configs, configs, "Configuration for subdomain simulators.")
    return cfg
end


function Jutul.select_linear_solver(sim::PArraySimulator)
    solver = Jutul.select_linear_solver(sim.storage.model)
    if !isa(solver, GenericKrylov)
        p = ILUZeroPreconditioner()
        solver = GenericKrylov(:bicgstab, preconditioner = p)
    end
    return solver
end

function Jutul.set_default_tolerances(sim::PArraySimulator; kwarg...)
    # Should automatically get added for subsimulators
end

function Jutul.initialize_before_first_timestep!(psim::PArraySimulator, first_dT; kwarg...)
    Jutul.@tic "solve" begin
        Jutul.@tic "secondary variables" map(psim.storage[:simulators]) do sim
            Jutul.update_secondary_variables!(sim.storage, sim.model)
            nothing
        end
    end
end

function Jutul.check_forces(psim::PArraySimulator, forces::AbstractVector, timesteps; per_step = false)
    map(psim.storage[:simulators], forces) do sim, f
        Jutul.check_forces(sim, f, timesteps; per_step = per_step)
    end
    nothing
end

function Jutul.reset_state_to_previous_state!(psim::PArraySimulator)
    map(psim.storage[:simulators]) do sim
        Jutul.reset_state_to_previous_state!(sim)
        nothing
    end
end

function Jutul.update_before_step!(psim::PArraySimulator, dt, forces; kwarg...)
    map(psim.storage[:simulators], forces) do sim, f
        Jutul.update_before_step!(sim, dt, f; kwarg...)
        nothing
    end
end

function Jutul.update_after_step!(psim::PArraySimulator, dt, forces; kwarg...)
    map(psim.storage[:simulators], forces) do sim, f
        Jutul.update_after_step!(sim, dt, f; kwarg...)
        nothing
    end
end

function Jutul.get_output_state(psim::PArraySimulator)
    state = map(psim.storage.simulators) do sim
        Jutul.get_output_state(sim.storage, sim.model)
    end
    return state
end

function Jutul.store_output!(states, reports, step, psim::PArraySimulator, config, report)
    subsims = psim.storage.simulators
    subconfigs = config[:configs]
    map(subsims, subconfigs) do sim, cfg
        Jutul.store_output!(states, reports, step, sim, cfg, report)
    end
end


function Jutul.forces_for_timestep(psim::PArraySimulator, forces::Vector, timesteps, step_index; per_step = true)
    force_for_step = map(psim.storage.simulators, forces) do sim, f
        if per_step
            f = f[step_index]
        end
        Jutul.forces_for_timestep(sim, f, timesteps, step_index)
    end
    return force_for_step
end

function Jutul.forces_for_timestep(psim::PArraySimulator, forces::Union{MPIArray, DebugArray, AbstractDict, Nothing, NamedTuple}, timesteps, step_index; per_step = false)
    subforces = map(psim.storage.simulators, forces) do sim, f
        Jutul.forces_for_timestep(sim, f, timesteps, step_index)
    end
    return subforces
end

function Jutul.perform_step!(simulator::PArraySimulator, dt, forces, config; iteration = 1, vararg...)
    s = simulator.storage
    tmr = s.global_timer
    simulators = s.simulators
    np = s.number_of_processes
    verbose = s.verbose
    configs = config[:configs]
    tic!(tmr)
    out = map(simulators, configs, forces) do sim, config, f
        e, conv, report = perform_step!(sim, dt, f, config; iteration = iteration, solve = false, update_secondary = true)
        (e, Int(conv), report)
    end
    errors, converged, reports = tuple_of_arrays(out)
    toc!(tmr, "Assembly")

    nconverged = sum(converged)
    all_processes_converged = nconverged == np
    max_error = reduce(max, errors, init = 0)

    if verbose && config[:info_level] > 1
        Jutul.jutul_message("It $(iteration-1)", "$nconverged/$np processes converged.")
    end
    report = Jutul.setup_ministep_report()
    map(reports) do subrep
        for k in [:secondary_time, :equations_time, :linear_system_time, :convergence_time]
            report[k] += subrep[k]
        end
    end
    # Proceed to linear solve
    if !all_processes_converged
        t_solved = @elapsed ok, n, res = parray_linear_solve!(simulator, config[:linear_solver])
        t_update = @elapsed map(simulators) do sim
            Jutul.update_primary_variables!(sim.storage, sim.model)
        end

        report[:linear_solve_time] = t_solved
        report[:update_time] = t_update
        report[:linear_iterations] = n
    end
    return (max_error, all_processes_converged, report)
end

function Jutul.post_update_linearized_system!(linearized_system, executor::PArrayExecutor, storage, model)
    lsys = linearized_system[1, 1]
    r = lsys.r
    n_self = executor.data[:n_self]
    unit_diagonalize!(r, lsys.jac, n_self)
end

function Jutul.retrieve_output!(sim::PArraySimulator, states, reports, config, n)
    np = sim.storage[:number_of_processes]
    is_main = sim.storage[:is_main_process]
    pth = config[:output_path]
    if np > 1 && is_main && pth isa String
        consolidate_distributed_results_on_disk!(pth, np, 1:n, cleanup = true)
    end
    Jutul.retrieve_output!(states, reports, config, n)
end