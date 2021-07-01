export LinearizedSystem, solve!, AMGSolver, CuSparseSolver, transfer, BlockDQGMRES, LUSolver

using SparseArrays, LinearOperators, StaticArrays
using IterativeSolvers, Krylov, AlgebraicMultigrid
using CUDA, CUDA.CUSPARSE

struct LinearizedSystem{L}
    jac
    r
    dx
    jac_buffer
    r_buffer
    dx_buffer
    matrix_layout::L
end

function LinearizedSystem(sparse_arg, context, layout)
    I, J, V, n, m = sparse_arg
    @assert n == m "Expected square system. Recieved $n (eqs) by $m (variables)."
    r = zeros(n)
    dx = zeros(n)
    jac = sparse(I, J, V, n, m)

    jac_buf, dx_buf, r_buf = get_nzval(jac), dx, r

    return LinearizedSystem(jac, r, dx, jac_buf, r_buf, dx_buf, layout)
end

function get_mul!(lsys)
    return mul!
end

function get_linear_operator(sys)
    apply! = get_mul!(sys)
    n = length(sys.r_buffer)
    return LinearOperator(Float64, n, n, false, false, apply!)
end


@inline function get_nzval(jac)
    return jac.nzval
end

function block_size(lsys::LinearizedSystem) 1 end

function solve!(sys::LinearizedSystem, linsolve::Nothing)
    solve!(sys)
end

function solve!(sys)
    if length(sys.dx) > 50000
        error("System too big for default direct solver.")
    end
    J = sys.jac
    r = sys.r
    sys.dx .= -(J\r)
    @assert all(isfinite, sys.dx) "Linear solve resulted in non-finite values."
end

