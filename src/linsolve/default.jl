export LinearizedSystem, solve!, AMGSolver, CuSparseSolver, transfer, LUSolver

using SparseArrays, LinearOperators, StaticArrays
using IterativeSolvers, Krylov, AlgebraicMultigrid
using CUDA, CUDA.CUSPARSE

abstract type TervLinearSystem end

struct LinearizedSystem{L} <: TervLinearSystem
    jac
    r
    dx
    jac_buffer
    r_buffer
    dx_buffer
    matrix_layout::L
end

struct LinearizedBlock{L} <: TervLinearSystem
    jac
    jac_buffer
    matrix_layout::L
    function LinearizedBlock(sparse_arg, context, layout)
        jac, jac_buf = build_jacobian(sparse_arg, context, layout)
        new{typeof(layout)}(jac, jac_buf, layout)
    end
end


LinearizedType = Union{LinearizedSystem, LinearizedBlock}
LSystem = Union{LinearizedType, Matrix{LinearizedSystem}}

function LinearizedSystem(sparse_arg, context, layout; r = nothing, dx = nothing)
    jac, jac_buf = build_jacobian(sparse_arg, context, layout)
    n, m = size(jac)
    @assert n == m "Expected square system. Recieved $n (eqs) by $m (primary variables)."
    dx, dx_buf = get_jacobian_vector(n, context, layout, dx)
    r, r_buf = get_jacobian_vector(n, context, layout, r)

    return LinearizedSystem(jac, r, dx, jac_buf, r_buf, dx_buf, layout)
end

function build_jacobian(sparse_arg, context, layout)
    I, J, V, n, m = sparse_arg
    jac = sparse(I, J, V, n, m)
    return (jac, get_nzval(jac))
end

function get_jacobian_vector(n, context, layout, v = nothing)
    if isnothing(v)
        v = zeros(n)
    end
    return (v, v)
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

function block_size(lsys::LSystem) 1 end

function solve!(sys::LSystem, linsolve, model = nothing, storage = nothing, dt = nothing)
    solve!(sys, linsolve)
end

function solve!(sys::LSystem, linsolve::Nothing)
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

