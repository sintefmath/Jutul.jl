export LinearizedSystem, solve!, AMGSolver, CuSparseSolver, transfer, LUSolver

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

function LinearizedSystem(sparse_arg, context, layout; allocate_r = true)
    I, J, V, n, m = sparse_arg
    @assert n == m "Expected square system. Recieved $n (eqs) by $m (variables)."
    if allocate_r
        r = zeros(n)
    else
        r = nothing
    end
    dx = zeros(n)
    jac = sparse(I, J, V, n, m)

    jac_buf, dx_buf, r_buf = get_nzval(jac), dx, r

    return LinearizedSystem(jac, r, dx, jac_buf, r_buf, dx_buf, layout)
end

function linear_operator(sys)
    if block_size(sys) == 1
        op = LinearOperator(sys.jac)
    else
        apply! = get_mul!(sys)
        n = length(sys.r_buffer)
        op = LinearOperator(Float64, n, n, false, false, apply!)
    end
    return op
end

function vector_residual(sys)
    return sys.r
end

function update_dx_from_vector!(sys, dx)
    sys.dx .= -dx
end

@inline function get_nzval(jac)
    return jac.nzval
end

function block_size(lsys::LinearizedSystem) 1 end

function solve!(sys::LinearizedSystem, linsolve, model = nothing, storage = nothing, dt = nothing)
    solve!(sys, linsolve)
end

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

