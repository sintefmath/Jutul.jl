export LinearizedSystem, MultiLinearizedSystem, linear_solve!, AMGSolver, CuSparseSolver, transfer, LUSolver

# using CUDA, CUDA.CUSPARSE

mutable struct FactorStore
    factor
    function FactorStore()
        new(nothing)
    end
end

function update_preconditioner!(f::FactorStore, g, g!, A, executor)
    if isnothing(f.factor)
        f.factor = g(A)
    else
        g!(f.factor, A)
    end
    return f.factor
end

function update_preconditioner!(f::FactorStore, g, g!, A::AbstractArray, executor)
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
end

function MultiLinearizedSystem(subsystems, context, layout; r = nothing, dx = nothing, reduction = nothing)
    n = 0
    schur_buffer = []
    #if represented_as_adjoint(layout)
    #    urng = 1:size(subsystems, 2)
    #else
        urng = 1:size(subsystems, 1)
    #end
    for i in urng
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
    MultiLinearizedSystem{typeof(layout)}(subsystems, r, dx, r_buf, dx_buf, reduction, FactorStore(), layout, schur_buffer)
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
        N = size(Jt, 1)^2
        V_buf = unsafe_reinterpret(Ft, nzval, length(nzval)*N)
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
            unsafe = true
            if unsafe
                # This is a bit dangerous but there are issues with performance
                # for reinterpreted arrays. We cast our local part into two "views"
                # by way of pointer trickery:
                # - One is a bz by n array of scalar type
                # - Another is a n array of the vector type
                ptr = Base.unsafe_convert(Ptr{Ft}, v)
                v_buf = Base.unsafe_wrap(Array, ptr, (bz, n))

                ptr = Base.unsafe_convert(Ptr{Vt}, v)
                v = Base.unsafe_wrap(Array, ptr, n)
            else
                v_buf = reshape(v, bz, :)
                v = reinterpret(reshape, Vt, v_buf)
            end
        end
    end
    return (v, v_buf)
end

jacobian(sys) = sys.jac
residual(sys) = sys.r
linear_system_context(model, sys) = model.context

function prepare_linear_solve!(sys)
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

function apply_left_diagonal_scaling!(M::SparseMatrixCSC{SMatrix{N, N, T, NN}, Int}, D::AbstractVector) where {N, T, NN}
    n = length(D)
    nrow, ncol = size(M)
    nzval = nonzeros(M)
    rows = rowvals(M)

    D_mat = MMatrix{N, N, T, NN}(I)
    for col in 1:ncol
        for pos in nzrange(M, col)
            row = rows[pos]
            M_ij = nzval[pos]
            for k in 1:N
                D_mat[k, k] = D[(row-1)*N + k]
            end
            nzval[pos] = D_mat*nzval[pos]
        end
    end
    return M
end

function apply_left_diagonal_scaling!(M::SparseMatrixCSC, D::AbstractVector)
    n = length(D)
    nrow, ncol = size(M)
    nzval = nonzeros(M)
    rows = rowvals(M)
    for col in 1:ncol
        for pos in nzrange(M, col)
            row = rows[pos]
            nzval[pos] = D[row]*nzval[pos]
        end
    end
    return M
end

function apply_left_diagonal_scaling!(M::AbstractVector, D::AbstractVector)
    @assert length(M) == length(D)
    for i in eachindex(M)
        M[i] = D[i]*M[i]
    end
    return M
end

function apply_scaling_to_linearized_system!(lsys::LinearizedSystem, F, type::Symbol)
    if type == :diagonal
        if isnothing(F)
            T = eltype(lsys.dx_buffer)
            n_diag = length(lsys.dx_buffer)
            F = zeros(T, n_diag)
        end
        F = diagonal_inverse_scaling!(lsys, F)
        apply_left_diagonal_scaling!(lsys.jac, F)
        apply_left_diagonal_scaling!(vec(lsys.r_buffer), F)
    else
        @assert type == :none
    end
    return (lsys, F)
end

function apply_scaling_to_linearized_system!(lsys::LinearizedBlock, F, type::Symbol)
    if type == :diagonal
        @assert !isnothing(F)
        apply_left_diagonal_scaling!(lsys.jac, F)
    end
    return (lsys, F)
end

function diagonal_inverse_scaling!(lsys::LinearizedSystem, F)
    J = lsys.jac
    return diagonal_inverse_scaling!(J, F)
end

function diagonal_inverse_scaling!(A::AbstractSparseMatrix{T, Int}, F) where T<:StaticMatrix
    n = size(A, 1)
    m = length(F)
    bz = m ÷ n
    @assert bz > 0
    for i in 1:n
        A_ii = A[i, i]
        for j in 1:bz
            F[(i-1)*bz + j] = inv(A_ii[j, j])
        end
    end
    return F
end

function diagonal_inverse_scaling!(A::AbstractSparseMatrix, F)
    T = eltype(A)
    for i in eachindex(F)
        A_ii = A[i, i]
        if A_ii ≈ 0
            A_ii = 1.0
        else
            A_ii = 1.0/A_ii
        end
        F[i] = A_ii
    end
    return F
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

function update_dx_from_vector!(sys, dx_from_solver; dx = sys.dx)
    dx .= -dx_from_solver
end

block_size(lsys::LSystem) = 1

function linear_solve_return(ok = true, iterations = 1, stats = nothing; prepare = 0.0)
    (ok = ok, iterations = iterations, stats = (stats = deepcopy(stats), prepare = prepare))
end

function linear_solve!(sys, ::Nothing, arg...; dx = sys.dx, r = sys.r, atol = nothing, rtol = nothing, executor = default_executor())
    limit = 100_000
    n = length(sys.dx)
    if n > limit
        error("System too big for default direct solver. (Limit is $limit, system was $n by $n.")
    end
    dx .= -(sys.jac\r)
    @assert all(isfinite, dx) "Linear solve resulted in non-finite values."
    return linear_solve_return()
end

