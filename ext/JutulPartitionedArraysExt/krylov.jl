function parray_linear_solve!(simulator, lsolve; 
        atol = Jutul.linear_solver_tolerance(lsolve, :absolute),
        rtol = Jutul.linear_solver_tolerance(lsolve, :relative),
    )
    s = simulator.storage
    cfg = lsolve.config
    simulators = s.simulators
    b = s.distributed_residual
    @tic "prepare" prepare_distributed_solve!(simulators, b)
    bsolver = bsolver_setup!(lsolve, simulators, b)
    @assert lsolve.solver == :bicgstab "Only :bicgstab supported, was $(lsolve.solver)"

    return inner_krylov(bsolver, lsolve, simulator, simulators, cfg, b, s.verbose, atol, rtol)
end


function prepare_distributed_solve!(simulators, b)
    map(simulators, local_values(b)) do sim, r
        lsys = sim.storage.LinearizedSystem
        N = sim.executor.data[:n_self]
        Jutul.prepare_linear_solve!(lsys)
        r_sim = Jutul.vector_residual(lsys)
        @. r = r_sim
        nothing
    end
end

function bsolver_setup!(lsolve, simulators, b)
    if isnothing(lsolve.storage)
        lsolve.storage = local_bicgstab_solver(b)
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
    op = Jutul.parray_linear_system_operator(simulators, b)
    P = parray_preconditioner_linear_operator(simulator, lsolve, b)
    consistent!(b) |> wait

    max_it = cfg.max_iterations
    @tic "solve" Krylov.bicgstab!(
        bsolver, op, b,
        M = P,
        verbose = 0*Int(verbose),
        itmax = max_it,
        history = true,
        rtol = rtol,
        atol = atol
    )
    consistent!(bsolver.x) |> wait

    stats = bsolver.stats
    res = stats.residuals
    n_lin_its = length(res) - 1
    solved = stats.solved

    map(simulators, local_values(bsolver.x)) do sim, dx
        sys = sim.storage.LinearizedSystem
        Jutul.update_dx_from_vector!(sys, dx)
    end

    if verbose
        msg = stats.status
        n_lin_its == max_it
        initial_res = res[1]
        final_res = res[end]
        if !stats.solved
            @warn "Linear solver: $msg, final residual: $final_res, rel. value $(final_res/initial_res). rtol = $rtol, atol = $atol, max_it = $max_it"
        end
    end
    return Jutul.linear_solve_return(solved, n_lin_its, stats)
end

function local_bicgstab_solver(X::S) where S
    n = length(X)
    m = n
    FC = eltype(S)
    T  = real(FC)
    Δx = similar(X)
    x  = similar(X)
    r  = similar(X)
    p  = similar(X)
    v  = similar(X)
    s  = similar(X)
    qd = similar(X)
    yz = similar(X)
    t  = similar(X)
    stats = Krylov.SimpleStats(0, false, false, T[], T[], T[], "unknown")
    solver = Krylov.BicgstabSolver{T,FC,S}(m, n, Δx, x, r, p, v, s, qd, yz, t, false, stats)
    return solver
end
