export GenericKrylov

mutable struct GenericKrylov
    solver
    preconditioner
    config::IterativeSolverConfig
    function GenericKrylov(solver = dqgmres; preconditioner = nothing, kwarg...)
        new(solver, preconditioner, IterativeSolverConfig(;kwarg...))
    end
end

function atol(cfg)
    tol = cfg.absolute_tolerance
    return isnothing(tol) ? 0.0 : Float64(tol)
end

function rtol(cfg)
    tol = cfg.relative_tolerance
    return isnothing(tol) ? 0.0 : Float64(tol)
end

function verbose(cfg)
    return Int64(cfg.verbose)
end

function preconditioner(krylov::GenericKrylov, sys, arg...)
    M = krylov.preconditioner
    if isnothing(M)
        op = I
    else
        update!(M, sys)
        op = linear_operator(M, arg...)
    end
    return op
end

function solve!(sys::LSystem, krylov::GenericKrylov)
    solver, cfg = krylov.solver, krylov.config

    prepare_solve!(sys)
    r = vector_residual(sys)
    op = linear_operator(sys)

    L = preconditioner(krylov, sys, :left)
    R = preconditioner(krylov, sys, :right)
    v = verbose(cfg)
    max_it = cfg.max_iterations
    rt = rtol(cfg)
    at = atol(cfg)
    (x, stats) = solver(op, r, 
                            itmax = max_it,
                            verbose = v,
                            rtol = rt,
                            history = v > 0,
                            atol = at,
                            M = L, N = R)
    if !stats.solved
        @warn "Linear solve did not converge: $(stats.status). rtol = $rt, atol = $at, max_it = $max_it"
    end
    if v > 0
        r = stats.residuals
        @debug "Final residual $(r[end]), rel. value $(r[end]/r[1]) after $(length(r)-1) iterations."
    end
    update_dx_from_vector!(sys, x)
end
