export LinearizedSystem, MultiLinearizedSystem, solve!, AMGSolver, CuSparseSolver, transfer, LUSolver

# using CUDA, CUDA.CUSPARSE

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

function update!(f::FactorStore, g, g!, A::AbstractArray)
    if isnothing(f.factor)
        f.factor = map(g, A)
    else
        for (F, A_i) in zip(f.factor, A)
            g!(F, A_i)
        end
    end
    return f.factor
end

abstract type JutulLinearSystem end

struct LinearizedSystem{L, J, V, Z, B} <: JutulLinearSystem
    jac::J
    r::V
    dx::V
    jac_buffer::Z
    r_buffer::B
    dx_buffer::B
    matrix_layout::L
end

struct LinearizedBlock{R, C, J, B} <: JutulLinearSystem
    jac::J
    jac_buffer::B
    rowcol_block_size::NTuple{2, Int}
    function LinearizedBlock(sparse_arg, context, layout_row, layout_col, rowcol_dim)
        jac, jac_buf = build_jacobian(sparse_arg, context, layout_row, layout_col)
        new{typeof(layout_row), typeof(layout_col), typeof(jac), typeof(jac_buf)}(jac, jac_buf, rowcol_dim)
    end
end

function LinearizedBlock(A, bz::Tuple, row_layout, col_layout)
    pattern = to_sparse_pattern(A)
    context = DefaultContext(matrix_layout = row_layout)
    sys = LinearizedBlock(pattern, context, row_layout, col_layout, bz)
    J = sys.jac
    for (i, j) in zip(pattern.I, pattern.J)
        J[i, j] = 0.0
    end
    return sys
end

row_block_size(b::LinearizedBlock) = b.rowcol_block_size[1]
col_block_size(b::LinearizedBlock) = b.rowcol_block_size[2]

LinearizedType = Union{LinearizedSystem, LinearizedBlock}

struct MultiLinearizedSystem{L} <: JutulLinearSystem
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
            if i == 1
                push!(schur_buffer, zeros(ni*bz))
            else
                # Need two buffers of same size for Schur complement
                b = (zeros(ni*bz), zeros(ni*bz))
                push!(schur_buffer, b)
            end
            n += ni
        end
        schur_buffer = Tuple(schur_buffer)
        dx, dx_buf = get_jacobian_vector(n, context, layout, dx)
        r, r_buf = get_jacobian_vector(n, context, layout, r)
        new{typeof(layout)}(subsystems, r, dx, r_buf, dx_buf, reduction, FactorStore(), layout, schur_buffer)
    end
end

LSystem = Union{LinearizedType, MultiLinearizedSystem}

function LinearizedSystem(sparse_arg, context, layout; r = nothing, dx = nothing)
    jac, jac_buf, bz = build_jacobian(sparse_arg, context, layout)
    n, m = size(jac)
    if n != m
        @debug "Expected square system. Recieved $n (eqs) by $m (primary variables). Unless this is an adjoint system, something might be wrong."
    end
    if represented_as_adjoint(matrix_layout(context))
        nrows = m
        ncols = n
    else
        nrows = n
        ncols = m
    end
    dx, dx_buf = get_jacobian_vector(ncols, context, layout, dx, bz[1])
    r, r_buf = get_jacobian_vector(nrows, context, layout, r, bz[1])
    return LinearizedSystem(jac, r, dx, jac_buf, r_buf, dx_buf, layout)
end

function LinearizedSystem(A, r = nothing)
    pattern = to_sparse_pattern(A)
    layout = matrix_layout(A)
    context = DefaultContext(matrix_layout = layout)
    sys = LinearizedSystem(pattern, context, layout, r = r)
    J = sys.jac
    for (i, j, entry) in zip(findnz(A)...)
        J[i, j] = entry
    end
    return sys
end


function build_jacobian(sparse_arg, context, layout_row, layout_col = layout_row)
    # @assert sparse_arg.layout == layout
    I, J, n, m = ijnm(sparse_arg)
    bz = block_size(sparse_arg)
    Jt = jacobian_eltype(context, layout_row, bz)
    @assert Jt == jacobian_eltype(context, layout_col, bz)
    Ft = float_type(context)

    V = zeros(Jt, length(I))
    jac = build_sparse_matrix(context, I, J, V, n, m)
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
linear_system_context(model, sys) = model.context

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

function linear_operator(block::LinearizedBlock; skip_red = false)
    return LinearOperator(block.jac)
end

# function linear_operator(block::LinearizedBlock{EquationMajorLayout, BlockMajorLayout}; skip_red = false)
#     # Matrix is equation major.
#     # Row (and output) is equation major.
#     # Column (and vector we multiply with) is block major
#     jac = block.jac
#     function apply!(res, x, α, β::T) where T
#         @tic "spmv" begin
#             mul!(res, jac, x, α, β)
#         end
#         return res
#     end
#     n, m = size(jac)
#     return LinearOperator(Float64, n, m, false, false, apply!)
# end

# function linear_operator(block::LinearizedBlock{BlockMajorLayout, EquationMajorLayout}; skip_red = false)
#     # Handle this case specifically:
#     # Matrix is equation major.
#     # Row (and output) is block major.
#     # Column (and vector we multiply with) is equation major.
#     bz = row_block_size(block)
#     jac = block.jac

#     function apply!(res, x, α, β::T) where T
#         # mul!(C, A, B, α, β) -> C
#         # A*B*α + C*β
#         error()
#         @tic "spmv" begin
#             tmp_cell_major = jac*x
#             if α != one(T)
#                 lmul!(α, tmp_cell_major)
#             end
#             tmp_block_major = equation_major_to_block_major_view(tmp_cell_major, bz)

#             if β == zero(T)
#                 @. res = tmp_block_major
#             else
#                 if β != one(T)
#                     lmul!(β, res)
#                 end
#                 @. res += tmp_block_major
#             end
#         end
#         return res
#     end
#     n, m = size(jac)
#     return LinearOperator(Float64, n, m, false, false, apply!)
# end

function vector_residual(sys::Vector)
    return map(x -> x.r, sys)
end

function vector_residual(sys)
    return sys.r
end

function update_dx_from_vector!(sys, dx)
    sys.dx .= -dx
end

block_size(lsys::LSystem) = 1

linear_solve_return(ok = true, iterations = 1, stats = nothing) = (ok = ok, iterations = iterations, stats = stats)

solve!(sys::LSystem, linsolve, model, storage = nothing, dt = nothing, recorder = nothing) = solve!(sys, linsolve)
solve!(sys::LSystem, linsolve::Nothing) = solve!(sys)

function solve!(sys; dx = sys.dx, r = sys.r, J = sys.jac)
    limit = 50000
    n = length(sys.dx)
    if n > limit
        error("System too big for default direct solver. (Limit is $limit, system was $n by $n.")
    end
    dx .= -(J\r)
    @assert all(isfinite, dx) "Linear solve resulted in non-finite values."
    return linear_solve_return()
end

