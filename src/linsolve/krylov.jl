import LinearAlgebra.mul!

export GenericKrylov

struct PrecondWrapper{T, K}
    op::T
    x::K
    function PrecondWrapper(op::T_o) where T_o<:LinearOperator
        x = zeros(eltype(op), size(op, 1))
        T_x = typeof(x)
        new{T_o, T_x}(op, x)
    end
end

Base.eltype(p::PrecondWrapper) = eltype(p.op)

LinearAlgebra.mul!(x, p::PrecondWrapper, y) = mul!(x, p.op, y)
LinearAlgebra.mul!(x, p::PrecondWrapper, y, α, β) = mul!(x, p.op, y, α, β)

function LinearAlgebra.ldiv!(p::PrecondWrapper, x)
    y = p.x
    y = copy!(y, x)
    mul!(x, p.op, y) 
end

mutable struct GenericKrylov
    solver
    preconditioner
    storage
    x
    r_norm
    config::IterativeSolverConfig
    function GenericKrylov(solver = :gmres; preconditioner = nothing, kwarg...)
        new(solver, preconditioner, nothing, nothing, nothing, IterativeSolverConfig(;kwarg...))
    end
end

function atol(cfg, T = Float64)
    tol = cfg.absolute_tolerance
    return T(isnothing(tol) ? 0.0 : tol)
end

function rtol(cfg, T = Float64)
    tol = cfg.relative_tolerance
    return T(isnothing(tol) ? 0.0 : tol)
end

function verbose(cfg)
    return Int64(cfg.verbose)
end

function preconditioner(krylov::GenericKrylov, sys, model, storage, recorder, side, arg...)
    M = krylov.preconditioner
    if isnothing(M)
        op = I
    else
        op = PrecondWrapper(linear_operator(M, side, arg...))
    end
    return op
end

export update!
function update_preconditioner!(prec, sys, model, storage, recorder)
    update!(prec, sys, model, storage, recorder)
end

function solve!(sys::LSystem, krylov::GenericKrylov, model, storage = nothing, dt = nothing, recorder = nothing)
    solver = krylov.solver
    cfg = krylov.config
    prec = krylov.preconditioner
    Ft = float_type(model.context)
    @timeit "prepare" prepare_solve!(sys)
    r = vector_residual(sys)
    op = linear_operator(sys)
    @timeit "precond" update_preconditioner!(prec, sys, model, storage, recorder)
    L = preconditioner(krylov, sys, model, storage, recorder, :left, Ft)
    R = preconditioner(krylov, sys, model, storage, recorder, :right, Ft)
    v = verbose(cfg)
    max_it = cfg.max_iterations
    rt = rtol(cfg, Ft)
    at = atol(cfg, Ft)

    rtol_nl = cfg.nonlinear_relative_tolerance    
    if !isnothing(recorder) && !isnothing(rtol_nl)
        it = subiteration(recorder)
        rtol_relaxed = cfg.relaxed_relative_tolerance
        r_k = norm(r)
        r_0 = krylov.r_norm
        if it == 1
            krylov.r_norm = r_k
        elseif !isnothing(rtol_nl) && !isnothing(r_0)
            maybe_rtol = r_0*rtol_nl/r_k
            rt = max(min(maybe_rtol, rtol_relaxed), rt)
        end
    end
    if solver == :gmres
        if isnothing(krylov.storage)
            krylov.storage = GmresSolver(op, r, 20)
        end
        F = krylov.storage
        solve_f = (arg...; kwarg...) -> Krylov.gmres!(F, arg...; reorthogonalization = true, kwarg...)
        in_place = true
    elseif solver == :bicgstab
        if isnothing(krylov.storage)
            krylov.storage = BicgstabSolver(op, r)
        end
        F = krylov.storage
        solve_f = (arg...; kwarg...) -> Krylov.bicgstab!(F, arg...; kwarg...)
        in_place = true
    else
        solve_f = eval(:(Krylov.$solver))
        in_place = false
    end
    @timeit "solve" ret = solve_f(op, r, 
                            itmax = max_it,
                            verbose = v,
                            rtol = rt,
                            history = true,
                            atol = at,
                            M = L; cfg.arguments...)
    if in_place
        x, stats = (krylov.storage.x, krylov.storage.stats)
    else
        x, stats = ret
    end
    res = stats.residuals
    n = length(res) - 1
    solved = stats.solved
    msg = stats.status
    if n > 1
        initial_res = res[1]
        final_res = res[end]
    else
        initial_res = norm(r)
        final_res = norm(op*x - r)
    end

    if !solved && v >= 0
        @warn "Linear solver: $msg, final residual: $final_res, rel. value $(final_res/initial_res). rtol = $rt, atol = $at, max_it = $max_it, solver = $solver"
    elseif v > 0 
        @debug "$n lsolve its: Final residual $final_res, rel. value $(final_res/initial_res)."
    end

    @timeit "update dx" update_dx_from_vector!(sys, x)
    return linear_solve_return(solved, n, stats)
end

function is_mutating(f)
    return String(Symbol(f))[end] == '!'
end
