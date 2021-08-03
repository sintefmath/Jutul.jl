import LinearAlgebra.mul!

export GenericKrylov

struct PrecondWrapper
    op::LinearOperator
    x
    function PrecondWrapper(op)
        x = zeros(eltype(op), size(op, 1))
        new(op, x)
    end
end

Base.eltype(p::PrecondWrapper) = eltype(p.op)

LinearAlgebra.mul!(x, p::PrecondWrapper, arg...) = mul!(x, p.op, arg...)

function LinearAlgebra.ldiv!(p::PrecondWrapper, x)
    y = p.x
    y = copy!(y, x)
    mul!(x, p.op, y) 
end

mutable struct GenericKrylov
    solver
    preconditioner
    x
    config::IterativeSolverConfig
    function GenericKrylov(solver = gmres!; preconditioner = nothing, kwarg...)
        new(solver, preconditioner, nothing, IterativeSolverConfig(;kwarg...))
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
    solver = krylov.solver
    cfg = krylov.config
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
        if is_mutating(solver)
            if isnothing(krylov.x)
                krylov.x = similar(r)
            end
            x = krylov.x
            x .= zero(eltype(r))

            (x, history) = solver(x, op, r, initially_zero = true, abstol = at, reltol = rt, log = true, maxiter = max_it, verbose = v > 0, Pl = L)#, Pr = R)
        else
            (x, history) = solver(op, r, abstol = at, reltol = rt, log = true, maxiter = max_it, verbose = v > 0, Pl = L)#, Pr = R)
        end
        
        solved = history.isconverged
        msg = history
        n = history.iters
        res = history.data[:resnorm]
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
        solved = stats.solved
        msg = stats.status
    end
    if n > 1
        initial_res = res[1]
        final_res = res[end]
    else
        initial_res = 0
        final_res = 0
    end

    if !solved
        @warn "Linear solver: $msg, final residual: $final_res, rel. value $(final_res/initial_res). rtol = $rt, atol = $at, max_it = $max_it"
    end
    if v > 0 || true
        @debug "Final residual $final_res, rel. value $(final_res/initial_res) after $n iterations."
    end
    update_dx_from_vector!(sys, x)
end

function is_mutating(f)
    return String(Symbol(f))[end] == '!'
end
function from_IterativeSolvers(f)
    return f == gmres || f == gmres! || f == bicgstabl || f == bicgstabl!
end