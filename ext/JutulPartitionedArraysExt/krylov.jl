function parray_linear_solve!(
        simulator, lsolve;
        atol = Jutul.linear_solver_tolerance(lsolve, :absolute),
        rtol = Jutul.linear_solver_tolerance(lsolve, :relative),
    )
    s = simulator.storage
    cfg = lsolve.config
    simulators = s.simulators
    b = s.distributed_residual
    @tic "prepare" prepare_distributed_solve!(simulators, b)
    bsolver = bsolver_setup!(lsolve, simulators, b)
    @assert lsolve.solver in (:bicgstab, :gmres) "Only :bicgstab and :gmres supported, was $(lsolve.solver)"

    return inner_krylov(bsolver, lsolve, simulator, simulators, cfg, b, s.verbose, atol, rtol)
end


function prepare_distributed_solve!(simulators, b)
    return map(simulators, local_values(b)) do sim, r
        lsys = Jutul.get_simulator_storage(sim).LinearizedSystem
        N = sim.executor.data[:n_self]
        Jutul.prepare_linear_solve!(lsys)
        r_sim = Jutul.vector_residual(lsys)
        @. r = r_sim
        nothing
    end
end

function bsolver_setup!(lsolve, simulators, b)
    if isnothing(lsolve.storage)
        if lsolve.solver == :bicgstab
            lsolve.storage = local_bicgstab_solver(b)
        else
            lsolve.storage = local_gmres_solver(b)
        end
        prec = lsolve.preconditioner
        # Expand to one per local process
        if !isa(prec, Tuple)
            precs = map(simulators) do _
                deepcopy(prec)
            end
            lsolve.preconditioner = (prec, precs)
        end
    end
    bsolver = lsolve.storage
    @. bsolver.x = zero(eltype(bsolver.x))
    return bsolver
end


function inner_krylov(bsolver, lsolve, simulator, simulators, cfg, b, verbose, atol, rtol)
    t_op = @elapsed op = Jutul.parray_linear_system_operator(simulators, length(b))
    t_prec = @elapsed P = parray_preconditioner_linear_operator(simulator, lsolve, b)
    @tic "communication" consistent!(b) |> wait
    # TODO: Fix left preconditioning support

    max_it = cfg.max_iterations
    l_arg = (bsolver, op, b)
    l_kwarg = (
        M = P,
        verbose = 0 * Int(verbose),
        itmax = max_it,
        history = true,
        rtol = rtol,
        atol = atol,
    )
    if lsolve.solver == :bicgstab
        F! = Krylov.bicgstab!
        extra = NamedTuple()
    else
        F! = Krylov.gmres!
        extra = (restart = true,)
    end
    @tic "solve" F!(l_arg...; extra..., l_kwarg...)
    @tic "communication" consistent!(bsolver.x) |> wait

    stats = bsolver.stats
    res = stats.residuals
    n_lin_its = length(res) - 1
    solved = stats.solved

    @tic "dx update" map(simulators, local_values(bsolver.x)) do sim, dx
        sys = Jutul.get_simulator_storage(sim).LinearizedSystem
        Jutul.update_dx_from_vector!(sys, dx)
    end

    msg = stats.status
    n_lin_its == max_it
    initial_res = res[1]
    final_res = res[end]
    if !stats.solved
        bad_msg = "Linear solver: $msg, final residual: $final_res, rel. value $(final_res / initial_res). rtol = $rtol, atol = $atol, max_it = $max_it"
        if res[end] / res[1] > 1.0
            error(bad_msg)
        elseif verbose > 0
            @warn bad_msg
        end
    end
    t_prep = t_op + t_prec
    return Jutul.linear_solve_return(
        solved, n_lin_its, stats,
        prepare = t_prep,
        precond = P.time,
        precond_count = P.count
    )
end

function local_bicgstab_solver(X)
    return Krylov.BicgstabWorkspace(Krylov.KrylovConstructor(X))
end

function local_gmres_solver(X, memory = 20)
    return Krylov.GmresWorkspace(Krylov.KrylovConstructor(X); memory)
end
