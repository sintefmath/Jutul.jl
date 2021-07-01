export GenericKrylov

mutable struct GenericKrylov
    solver
    preconditioner
    config::IterativeSolverConfig
    function GenericKrylov(solver = dqgmres; preconditioner = nothing, kwarg...)
        new(solver, preconditioner, IterativeSolverConfig(kwarg...))
    end
end

function atol(cfg)
    tol = cfg.absolute_tolerance
    return isnothing(tol) ? 0.0 : tol
end

function rtol(cfg)
    tol = cfg.relative_tolerance
    return isnothing(tol) ? 0.0 : tol
end

function verbose(cfg)
    return Int64(cfg.verbose)
end

function preconditioner(krylov::GenericKrylov, sys)
    M = krylov.preconditioner
    if isnothing(M)
        op = I
    else
        update!(M, sys.jac, sys.r)
        op = linear_operator(M)
    end
    return op
end
function solve!(sys::LinearizedSystem, krylov::GenericKrylov)
    solver, cfg = krylov.solver, krylov.config

    r = vector_residual(sys)
    op = linear_operator(sys)

    M = preconditioner(krylov, sys)
    (x, stats) = solver(op, r, history = true, verbose = verbose(cfg), rtol = rtol(cfg), atol = atol(cfg), M = M)
    if !stats.solved
        @warn "Linear solve did not converge: $(stats.status)"
    end
    update_dx_from_vector!(sys, x)
end
