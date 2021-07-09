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

    r = vector_residual(sys)
    op = linear_operator(sys)

    L = preconditioner(krylov, sys, :left)
    R = preconditioner(krylov, sys, :right)
    v = verbose(cfg)
    (x, stats) = solver(op, r, 
                            itmax = cfg.max_iterations,
                            verbose = v,
                            rtol = rtol(cfg),
                            history = v > 0,
                            atol = atol(cfg),
                            M = L, N = R)
    if !stats.solved
        @warn "Linear solve did not converge: $(stats.status)"
    end
    if v > 0
        r = stats.residuals
        @debug "Final residual $(r[end]), rel. value $(r[end]/r[1]) after $(length(r)) iterations."
    end
    update_dx_from_vector!(sys, x)
end

# function get_mul!(sys::MultiLinearizedSystem)
#     subsystems = sys.subsystems
#     n, m = size(subsystems)
#     function block_mul!(res, x, α, β::T) where T
#         if β == zero(T)
#             row_offset = 0
#             for row = 1:n
#                 col_offset = 0
#                 for col = 1:m
#                     M = subsystems[row, col].jac
#                     nrows, ncols = size(M)
#                     rpos = (row_offset+1):(row_offset+nrows)
#                     cpos = (col_offset+1):(col_offset+ncols)
#                     # Grab views into the global vectors and applu matrix here
#                     res_v = view(res, rpos)
#                     x_v = view(x, cpos)
#                     mul!(res_v, M, x_v)
#                     if α != one(T)
#                         # TODO: This is wrong
#                         error("May be wrong.")
#                         lmul!(α, res_v)
#                     end
#                     col_offset += ncols
#                 end
#                 row_offset += size(subsystems[row, 1].jac, 1)
#             end
#         else
#             error("Not implemented yet.")
#         end
#     end
#     return block_mul!
# end

function linear_operator(sys::MultiLinearizedSystem)
    S = sys.subsystems
    d = size(S, 2)
    ops = map(linear_operator, permutedims(S))
    op = hvcat(d, ops...)
    return op
end
