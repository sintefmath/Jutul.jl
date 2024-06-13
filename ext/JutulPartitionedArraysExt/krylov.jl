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
    @assert lsolve.solver in (:bicgstab, :gmres) "Only :bicgstab and :gmres supported, was $(lsolve.solver)"

    return inner_krylov(bsolver, lsolve, simulator, simulators, cfg, b, s.verbose, atol, rtol)
end


function prepare_distributed_solve!(simulators, b)
    map(simulators, local_values(b)) do sim, r
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
        verbose = 0*Int(verbose),
        itmax = max_it,
        history = true,
        rtol = rtol,
        atol = atol
    )
    if lsolve.solver == :bicgstab
        F! = Krylov.bicgstab!
        extra = NamedTuple()
    else
        F! = Krylov.gmres!
        extra = (restart = true, )
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
        bad_msg = "Linear solver: $msg, final residual: $final_res, rel. value $(final_res/initial_res). rtol = $rtol, atol = $atol, max_it = $max_it"
        if res[end]/res[1] > 1.0
            error(bad_msg)
        elseif verbose > 0
            @warn bad_msg
        end
    end
    t_prep = t_op + t_prec
    return Jutul.linear_solve_return(solved, n_lin_its, stats, prepare = t_prep)
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
    stats = Krylov.SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
    solver = Krylov.BicgstabSolver{T,FC,S}(m, n, Δx, x, r, p, v, s, qd, yz, t, false, stats)
    return solver
end

function local_gmres_solver(X::S, memory = 20) where S
    n = length(X)
    m = n
    FC = eltype(S)
    T  = real(FC)
    Δx = similar(X)
    x  = similar(X)
    w  = similar(X)
    p  = similar(X)
    q  = similar(X)
    V = S[similar(X) for i = 1:memory]
    c = Vector{T}(undef, memory)
    s  = Vector{FC}(undef, memory)
    z  = Vector{FC}(undef, memory)
    R  = Vector{FC}(undef, div(memory * (memory+1), 2))
    stats = Krylov.SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
    solver = Krylov.GmresSolver{T,FC,S}(m, n, Δx, x, w, p, q, V, c, s, z, R, false, 0, stats)
    return solver
end
