
function Jutul.simulator_config(sim::PArraySimulator; extra_timing = false, info_level = 1, kwarg...)
    cfg = JutulConfig("Simulator config")
    v = sim.storage.verbose
    main_process_info_level = info_level
    if !v
        info_level = -1
    end
    is_mpi = sim.backend isa Jutul.MPI_PArrayBackend
    is_mpi_win = is_mpi && Sys.iswindows()
    extra_timing = extra_timing && v
    add_option!(cfg, :consolidate_results, true, "Consolidate states after simulation (serially).", types = Bool)
    add_option!(cfg, :delete_on_consolidate, true, "Delete processor states once consolidated.", types = Bool)
    add_option!(cfg, :info_level_parray, main_process_info_level, "Info level for outer printing", types = Int)

    cfg, unused = Jutul.simulator_config!(cfg, sim;
        kwarg...,
        info_level = info_level,
        ascii_terminal = is_mpi_win,
        extra_timing = extra_timing,
        output_unused = true
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
        subconfig = Jutul.simulator_config(sim;
            info_level = -1,
            extra_timing = extra_timing,
            output_path = pth,
            unused...
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
            s = Jutul.get_simulator_storage(sim)
            m = Jutul.get_simulator_model(sim)
            Jutul.update_secondary_variables!(s, m)
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
    # TODO: Fix output for this function.
    return missing
end

function Jutul.get_output_state(psim::PArraySimulator)
    state = map(psim.storage.simulators) do sim
        Jutul.get_output_state(sim.storage, sim.model)
    end
    return state
end

function Jutul.store_output!(states, reports, step, psim::PArraySimulator, config, report; substates = missing)
    subsims = psim.storage.simulators
    subconfigs = config[:configs]
    # TODO: Deal with substates for PArray.
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

function Jutul.perform_step!(
        simulator::PArraySimulator, dt, forces, config;
        solve = true,
        iteration = 1,
        executor = Jutul.default_executor(),
        vararg...
    )
    s = simulator.storage
    simulators = s.simulators
    np = s.number_of_processes
    configs = config[:configs]
    reports = map(simulators, configs, forces) do sim, config, f
        return Jutul.perform_step_per_process_initial_update!(
            sim, dt, f, config,
            update_secondary = true,
            executor = Jutul.simulator_executor(sim),
            iteration = iteration
        )
    end
    if haskey(s, :distributed_primary_variables)
        Jutul.parray_synchronize_primary_variables(simulator)
    end
    out = map(simulators, configs, forces, reports) do sim, config, f, rep
        t_s = get(rep, :secondary_time, 0.0)
        e, conv, rep = perform_step!(sim, dt, f, config;
            iteration = iteration,
            solve = false,
            report = rep,
            update_secondary = false,
            executor = Jutul.simulator_executor(sim),
            vararg...
        )
        if haskey(rep, :secondary_time)
            rep[:secondary_time] += t_s
        else
            rep[:secondary_time] = t_s
        end
        return (e, Int(conv))
    end
    errors, converged = tuple_of_arrays(out)

    nconverged = sum(converged)
    all_processes_converged = nconverged == np
    max_error = reduce(max, errors, init = 0)

    parray_print_convergence_status(simulator, config, reports, converged, iteration, config[:max_nonlinear_iterations], nconverged, np)
    should_solve = solve && !all_processes_converged
    report = Jutul.setup_ministep_report(converged = all_processes_converged, solved = should_solve)
    map(reports) do subrep
        for k in [:secondary_time, :equations_time, :linear_system_time, :convergence_time]
            report[k] += subrep[k]
        end
        nothing
    end
    # Proceed to linear solve
    if should_solve
        try
            t_solved = @elapsed ok, n, res = parray_linear_solve!(simulator, config[:linear_solver])
            t_update = @elapsed map(simulators) do sim
                Jutul.update_primary_variables!(
                    Jutul.get_simulator_storage(sim),
                    Jutul.get_simulator_model(sim)
                )
                nothing
            end
            report[:linear_solver] = res
            report[:linear_solve_time] = t_solved
            report[:update_time] = t_update
            report[:linear_iterations] = n
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
    report[:update] = missing
    return (max_error, all_processes_converged, report)
end

function parray_print_convergence_status(simulator, config, reports, converged, iteration, maxits, nconverged, np)
    s = simulator.storage
    info_level = config[:info_level_parray]
    if s.verbose && info_level > 1
        if nconverged == np
            msg = "All"
        else
            msg = "$nconverged/$np"
        end
        Jutul.jutul_message("It $(iteration-1)/$maxits", "$msg processes converged.", color = :cyan)
    end
    if info_level > 2
        # These get printed on all processes with a barrier. Performance cost to
        # barriers, and potentially a lot of output being printed.
        comm = s[:comm]
        rank_sz = MPI.Comm_size(comm)
        self_rank = MPI.Comm_rank(comm)+1
        for rank in 1:rank_sz
            MPI.Barrier(comm)
            sim_ctr = 1
            map(reports) do report
                if rank == self_rank
                    jutul_message("Process $rank/$rank_sz", "Convergence for simulator $sim_ctr", color = :light_cyan)
                    Jutul.get_convergence_table(report[:errors], info_level, iteration, config)
                    sim_ctr += 1
                end
            end
        end
    end
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
    comm = sim.storage[:comm]
    # Main processor is responsible for consolidating output.
    if config[:consolidate_results]
        MPI.Barrier(comm)
        if np > 1 && is_main && pth isa String
            Jutul.consolidate_distributed_results_on_disk!(pth, np, 1:n, cleanup = config[:delete_on_consolidate], verbose = config[:info_level] > 0)
        end
        MPI.Barrier(comm)
    end
    states, reports = Jutul.retrieve_output!(states, reports, config, n, read_states = config[:output_states] && is_main)
    return (states, reports)
end

function Jutul.simulator_reports_per_step(psim::PArraySimulator)
    if psim.backend isa MPI_PArrayBackend
        n = 0
        map(psim.storage[:simulators]) do s
            n += 1
        end
    else
        n = 1
    end
    return n
end
