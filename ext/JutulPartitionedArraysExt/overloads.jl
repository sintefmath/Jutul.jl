
function Jutul.simulator_config(sim::PArraySimulator; kwarg...)
    cfg = JutulConfig("Simulator config")
    v = sim.storage.verbose
    if v
        il = 1
    else
        il = -1
    end
    sel = Vector{Any}()
    t_base = TimestepSelector(initial_absolute = 3600.0*24.0, max = Inf)
    push!(sel, t_base)
    t_its = IterationTimestepSelector(8, offset = 1)
    push!(sel, t_its)

    # tol_cnv = 1e-3
    # tol_mb = 1e-7
    # tol_cnv_well = 10*tol_cnv
    # tol_mb_well = 1e4*tol_mb

    is_mpi_win = isa(sim, MPISimulator) && Sys.iswindows()
    extra_timing = v
    extra_timing = false
    Jutul.simulator_config!(cfg, sim; info_level = il, ascii_terminal = is_mpi_win, timestep_selectors = sel, extra_timing = extra_timing, kwarg...)
    simulators = sim.storage[:simulators]
    configs = map(simulators) do sim
        subconfig = Jutul.simulator_config(sim, info_level = -1, timestep_selectors = sel, extra_timing = extra_timing)
        # TODO: Fix this for JutulDarcy integration
        # JutulDarcy.set_default_cnv_mb!(subconfig, sim.model, tol_cnv = tol_cnv, tol_mb = tol_mb, tol_cnv_well = tol_cnv_well, tol_mb_well = tol_mb_well)
        subconfig
    end
    add_option!(cfg, :configs, configs, "Configuration for subdomain simulators.")
    return cfg
end


function Jutul.select_linear_solver(sim::PArraySimulator)
    p = ILUZeroPreconditioner()
    # TODO: Fix this to be easily overloadable
    # p = CPRPreconditioner()
    # p = CPRPreconditioner(BoomerAMGPreconditioner())
    # @info "CPR hypre" p
    return GenericKrylov(:bicgstab, preconditioner = p)
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

    if verbose && config[:info_level] > 0
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
