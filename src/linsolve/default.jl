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

struct LinearizedBlock{R, C} <: TervLinearSystem
    jac
    jac_buffer
    matrix_layout::R
    residual_layout::C
    function LinearizedBlock(sparse_arg, context, layout, layout_r)
        jac, jac_buf = build_jacobian(sparse_arg, context, layout)
        I = typeof(layout)
        J = typeof(layout_r)
        new{I, J}(jac, jac_buf, layout, layout_r)
    end
end

LinearizedType = Union{LinearizedSystem, LinearizedBlock}

struct MultiLinearizedSystem{L} <: TervLinearSystem
    subsystems::Matrix{LinearizedType}
    r
    dx
    r_buffer
    dx_buffer
    matrix_layout::L
    function MultiLinearizedSystem(subsystems, context, layout; r = nothing, dx = nothing)
        n = 0
        for i = 1:size(subsystems, 1)
            ni, mi = size(subsystems[i, i].jac)
            @assert ni == mi
            n += ni
        end
        dx, dx_buf = get_jacobian_vector(n, context, layout, dx)
        r, r_buf = get_jacobian_vector(n, context, layout, r)
        new{typeof(layout)}(subsystems, r, dx, r_buf, dx_buf, layout)
    end
end

LSystem = Union{LinearizedType, MultiLinearizedSystem}

function LinearizedSystem(sparse_arg, context, layout; r = nothing, dx = nothing)
    jac, jac_buf, bz = build_jacobian(sparse_arg, context, layout)
    n, m = size(jac)
    @assert n == m "Expected square system. Recieved $n (eqs) by $m (primary variables)."
    dx, dx_buf = get_jacobian_vector(n, context, layout, dx, bz[1])
    r, r_buf = get_jacobian_vector(n, context, layout, r, bz[1])

    return LinearizedSystem(jac, r, dx, jac_buf, r_buf, dx_buf, layout)
end

function build_jacobian(sparse_arg, context, layout)
    @assert sparse_arg.layout == layout
    I, J, n, m = ijnm(sparse_arg)
    bz = block_size(sparse_arg)
    Jt = jacobian_eltype(context, layout, bz)
    Ft = float_type(context)

    V = zeros(Jt, length(I))
    jac = sparse(I, J, V, n, m)
    nzval = get_nzval(jac)
    if Ft == Jt
        V_buf = nzval
    else
        V_buf = reinterpret(reshape, Ft, nzval)
    end
    return (jac, V_buf, bz)
end

function get_jacobian_vector(n, context, layout, v = nothing, bz = 1)
    Ft = float_type(context)
    Vt = r_eltype(context, layout, bz)

    if Ft == Vt
        if isnothing(v)
            v = zeros(Vt, n)
        end
        v_buf = v
    else
        if isnothing(v)
            # No vector given - allocate and re-interpret
            v_buf = zeros(Ft, bz, n)
            v = reinterpret(reshape, Vt, v_buf)
        else
            # Vector (of floats) was given. Use as buffer, reinterpret.
            v::AbstractVector{<:Ft}
            @assert length(v) == n*bz "Expected buffer size $n*$bz, was $(length(v))."
            v_buf = reshape(v, bz, :)
            v = reinterpret(reshape, Vt, v_buf)
        end
    end
    return (v, v_buf)
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

function solve!(sys::LSystem, linsolve, model, storage = nothing, dt = nothing)
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

