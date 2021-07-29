

export GenericKrylov

struct PrecondWrapper
    op::LinearOperator
end

eltype(p::PrecondWrapper) = eltype(p.op)

mul!(x, p::PrecondWrapper, arg...) = mul!(x, p.op, arg...)

function LinearAlgebra.ldiv!(p::PrecondWrapper, x)
    y = copy(x)
    mul!(x, p.op, y, 1, 0) 
    # lmul!(x, p.op, y)
end

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
        op = PrecondWrapper(linear_operator(M, arg...))
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
    if from_IterativeSolvers(solver)
        # Pl = krylov.preconditioner.factor
        (x, history) = solver(op, r, abstol = at, reltol = rt, log = true, Pl = L, maxiter = max_it, verbose = v > 0)#, Pl = L, Pr = R, )
        display(history)
    else
        (x, stats) = solver(op, r, 
                                itmax = max_it,
                                verbose = v,
                                rtol = rt,
                                history = true,
                                atol = at,
                                M = L, N = R)
        res = stats.residuals
        n = length(res) - 1
        if !stats.solved
            @warn "Linear solver: $(stats.status), final residual: $(res[end]). rtol = $rt, atol = $at, max_it = $max_it"
        end
        if v > 0
            @debug "Final residual $(res[end]), rel. value $(res[end]/res[1]) after $n iterations."
        end
    end
    update_dx_from_vector!(sys, x)
end


function from_IterativeSolvers(f)
    return f == gmres
end