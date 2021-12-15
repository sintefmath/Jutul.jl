export LinearizedSystem, solve!, AMGSolver, CuSparseSolver, transfer, LUSolver

using SparseArrays, LinearOperators, StaticArrays
using IterativeSolvers, Krylov, AlgebraicMultigrid
using CUDA, CUDA.CUSPARSE

mutable struct FactorStore
    factor
    function FactorStore()
        new(nothing)
    end
end

function update!(f::FactorStore, g, g!, A)
    if isnothing(f.factor)
        f.factor = g(A)
    else
        g!(f.factor, A)
    end
    return f.factor
end

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
    residual_block_size::Integer
    function LinearizedBlock(sparse_arg, context, layout, layout_r, residual_block_size)
        jac, jac_buf = build_jacobian(sparse_arg, context, layout)
        I = typeof(layout)
        J = typeof(layout_r)
        new{I, J}(jac, jac_buf, layout, layout_r, residual_block_size)
    end
end

LinearizedType = Union{LinearizedSystem, LinearizedBlock}

struct MultiLinearizedSystem{L} <: TervLinearSystem
    subsystems::Matrix{LinearizedType}
    r
    dx
    r_buffer
    dx_buffer
    reduction
    factor
    matrix_layout::L
    schur_buffer
    function MultiLinearizedSystem(subsystems, context, layout; r = nothing, dx = nothing, reduction = nothing)
        n = 0
        schur_buffer = []
        for i = 1:size(subsystems, 1)
            J = subsystems[i, i].jac
            ni, mi = size(J)
            @assert ni == mi
            e = eltype(J)
            if e <: Real
                bz = 1
            else
                bz = size(e, 1)
            end
            push!(schur_buffer, zeros(ni*bz))
            n += ni
        end
        dx, dx_buf = get_jacobian_vector(n, context, layout, dx)
        r, r_buf = get_jacobian_vector(n, context, layout, r)
        new{typeof(layout)}(subsystems, r, dx, r_buf, dx_buf, reduction, FactorStore(), layout, schur_buffer)
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
    nzval = nonzeros(jac)
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

jacobian(sys) = sys.jac
residual(sys) = sys.r

function prepare_solve!(sys)
    # Default is to do nothing.
end

function linear_operator(sys::LinearizedSystem; skip_red = false)
    if block_size(sys) == 1
        op = LinearOperator(sys.jac)
    else
        apply! = get_mul!(sys)
        n = length(sys.r_buffer)
        op = LinearOperator(Float64, n, n, false, false, apply!)
    end
    return op
end

function linear_operator(block::LinearizedBlock{T, T}; skip_red = false) where {T <: Any}
    return LinearOperator(block.jac)
end

function linear_operator(block::LinearizedBlock{EquationMajorLayout, BlockMajorLayout}; skip_red = false)
    # Handle this case specifically:
    # A linearized block at [i,j] where residual vector entry [j] is from a system
    # with block ordering, but row [i] has equation major ordering
    bz = block.residual_block_size
    jac = block.jac
    return linear_operator_mixed_block(jac, bz)
end

function linear_operator_mixed_block(jac, bz)
    function apply!(res, x, α, β::T) where T
        # Note: This is unfortunately allocating, but applying
        # with a simple view leads to degraded performance.
        x_v = collect(reshape(reshape(x, bz, :)', :))
        if β == zero(T)
            mul!(res, jac, x_v)
            if α != one(T)
                lmul!(α, res)
            end
        else
            error("Not implemented.")
        end
    end
    n, m = size(jac)
    op = LinearOperator(Float64, n, m, false, false, apply!)
    return op
end

function vector_residual(sys)
    return sys.r
end

function update_dx_from_vector!(sys, dx)
    sys.dx .= -dx
end

function block_size(lsys::LSystem) 1 end

function solve!(sys::LSystem, linsolve, model, storage = nothing, dt = nothing, recorder = nothing)
    solve!(sys, linsolve)
end

function solve!(sys::LSystem, linsolve::Nothing)
    solve!(sys)
end

function solve!(sys)
    limit = 50000
    n = length(sys.dx)
    if n > limit
        error("System too big for default direct solver. (Limit is $limit, system was $n by $n.")
    end
    J = sys.jac
    r = sys.r
    sys.dx .= -(J\r)
    @assert all(isfinite, sys.dx) "Linear solve resulted in non-finite values."
end

