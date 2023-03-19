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

"""
GenericKrylov(solver = :gmres; preconditioner = nothing; <kwarg>)

Solver that wraps `Krylov.jl` with support for preconditioning.
"""
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

function linear_solver_tolerance(k::GenericKrylov, arg...; kwarg...)
    return linear_solver_tolerance(k.config, arg...; kwarg...)
end

function Base.show(io::IO, krylov::GenericKrylov)
    rtol = linear_solver_tolerance(krylov, :relative)
    atol = linear_solver_tolerance(krylov, :absolute)
    print(io, "Generic Krylov using $(krylov.solver) (ϵₐ=$atol, ϵ=$rtol) with prec = $(typeof(krylov.preconditioner))")
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

function linear_solve!(sys::LSystem,
                krylov::GenericKrylov,
                model,
                storage = nothing,
                dt = nothing,
                recorder = ProgressRecorder();
                dx = sys.dx_buffer,
                r = vector_residual(sys),
                atol = linear_solver_tolerance(krylov, :absolute),
                rtol = linear_solver_tolerance(krylov, :relative),
                rtol_nl = linear_solver_tolerance(krylov, :nonlinear_relative),
                rtol_relaxed = linear_solver_tolerance(krylov, :relaxed_relative)
                )
    cfg = krylov.config
    prec = krylov.preconditioner
    Ft = float_type(model.context)
    @tic "prepare" prepare_linear_solve!(sys)
    op = linear_operator(sys)
    @tic "precond" update_preconditioner!(prec, sys, model, storage, recorder)
    L = preconditioner(krylov, sys, model, storage, recorder, :left, Ft)
    # R = preconditioner(krylov, sys, model, storage, recorder, :right, Ft)
    v = Int64(cfg.verbose)
    max_it = cfg.max_iterations
    min_it = cfg.min_iterations
    use_relaxed_tol = !isnothing(recorder) && !isnothing(rtol_nl)
    use_true_rel_norm = cfg.true_residual

    if use_relaxed_tol || use_true_rel_norm
        r_k = norm(r, 2)
        if use_true_rel_norm
            # Try to avoid relative reduction in preconditioned norm
            atol = atol + rtol*r_k
            rtol = 0.0
        end
        if use_relaxed_tol
            it = subiteration(recorder)
            r_k = norm(r)
            r_0 = krylov.r_norm
            if it == 1
                krylov.r_norm = r_k
            elseif !isnothing(rtol_nl) && !isnothing(r_0)
                maybe_rtol = r_0*rtol_nl/r_k
                rtol = max(min(maybe_rtol, rtol_relaxed), rtol)
            end
        end
    end

    if min_it > 1
        rel_tol = rtol
        abs_tol = atol
        callback = solver -> krylov_termination_criterion(solver, abs_tol, rel_tol, min_it)
        # Set to small numbers so the callback fully controls convergence checks
        atol = 1e-20
        rtol = 1e-20
        manual_conv = true
    else
        callback = solver -> false
        manual_conv = false
    end
    solve_f, F = krylov_jl_solve_function(krylov, op, r)
    @tic "solve" solve_f(F, op, r;
                            itmax = max_it,
                            verbose = v,
                            rtol = rtol,
                            atol = atol,
                            history = true,
                            callback = callback,
                            M = L,
                            cfg.arguments...)
    x, stats = (krylov.storage.x, krylov.storage.stats)
    res = stats.residuals
    n = length(res) - 1
    solved = stats.solved
    msg = stats.status
    if n > 1
        initial_res = res[1]
        final_res = res[end]
    else
        initial_res = norm(r, 2)
        final_res = norm(op*x - r, 2)
    end

    bad_auto = !manual_conv && !solved
    bad_manual = manual_conv && stats.niter == max_it
    if (bad_manual || bad_auto) && v >= 0
        @warn "Linear solver: $msg, final residual: $final_res, rel. value $(final_res/initial_res). rtol = $rtol, atol = $atol, max_it = $max_it"
    elseif v > 0 
        @debug "$n lsolve its: Final residual $final_res, rel. value $(final_res/initial_res)."
    end
    # @info "$n lsolve its: Final residual $final_res, rel. value $(final_res/initial_res)." res

    @tic "update dx" update_dx_from_vector!(sys, x, dx = dx)
    return linear_solve_return(solved, n, stats)
end

function krylov_termination_criterion(solver, atol, rtol, min_its)
    res = solver.stats.residuals
    tol = atol + rtol*res[1]
    ok_tol = res[end] <= tol
    ok_its = length(res) > min_its
    done = ok_tol && ok_its
    return done
end

function is_mutating(f)
    return String(Symbol(f))[end] == '!'
end

function krylov_jl_solve_function(krylov::GenericKrylov, op, r, solver = krylov.solver)
    # Some trickery to generically wrapping a Krylov.jl solver.
    if solver == :gmres
        if isnothing(krylov.storage)
            krylov.storage = GmresSolver(op, r)
        end
        solve_f = gmres!
    elseif solver == :bicgstab
        if isnothing(krylov.storage)
            krylov.storage = BicgstabSolver(op, r)
        end
        solve_f = bicgstab!
    elseif solver == :fgmres
        if isnothing(krylov.storage)
            krylov.storage = FgmresSolver(op, r)
        end
        solve_f = fgmres!
    else
        if isnothing(krylov.storage)
            F_sym = Krylov.KRYLOV_SOLVERS[solver]
            krylov.storage = eval(:($F_sym($op, $r)))
        end
        solver_string = Symbol("$(solver)!")
        solve_f = eval(:(Krylov.$solver_string))
    end
    return (solve_f, krylov.storage)
end
