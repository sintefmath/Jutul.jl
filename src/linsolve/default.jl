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

struct LinearizedBlock{M, R, C} <: TervLinearSystem
    jac
    jac_buffer
    matrix_layout::M
    rowcol_block_size::NTuple{2, Int}
    function LinearizedBlock(sparse_arg, context, layout, layout_row, layout_col, rowcol_dim)
        jac, jac_buf = build_jacobian(sparse_arg, context, layout)
        new{typeof(layout), typeof(layout_row), typeof(layout_col)}(jac, jac_buf, layout, rowcol_dim)
    end
end

row_block_size(b::LinearizedBlock) = b.rowcol_block_size[1]
col_block_size(b::LinearizedBlock) = b.rowcol_block_size[2]

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

function linear_operator(block::LinearizedBlock{T, T, T}; skip_red = false) where T
    return LinearOperator(block.jac)
end

function linear_operator(block::LinearizedBlock{EquationMajorLayout, EquationMajorLayout, BlockMajorLayout}; skip_red = false)
    # Matrix is equation major.
    # Row (and output) is equation major.
    # Column (and vector we multiply with) is block major
    bz = col_block_size(block)
    jac = block.jac
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
    return LinearOperator(Float64, n, m, false, false, apply!)
end

function linear_operator(block::LinearizedBlock{EquationMajorLayout, BlockMajorLayout, EquationMajorLayout}; skip_red = false)
    # Handle this case specifically:
    # Matrix is equation major.
    # Row (and output) is block major.
    # Column (and vector we multiply with) is equation major.
    bz = row_block_size(block)
    jac = block.jac

    function apply!(res, x, α, β::T) where T
        # mul!(C, A, B, α, β) -> C
        # A*B*α + C*β
        tmp_cell_major = jac*x
        if α != one(T)
            lmul!(α, tmp_cell_major)
        end
        tmp_block_major = equation_major_to_block_major_view(tmp_cell_major, bz)

        if β == zero(T)
            @. res = tmp_block_major
        else
            if β != one(T)
                lmul!(β, res)
            end
            @. res += tmp_block_major
        end
    end
    n, m = size(jac)
    return LinearOperator(Float64, n, m, false, false, apply!)
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

