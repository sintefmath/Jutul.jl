"""
    setup_smoother(A, config=SPAI0(); reuse=nothing)

Build a reusable smoother/preconditioner for `A`. Symbolic data and work
buffers are retained by the returned state. Pass a compatible state through
`reuse` to update coefficients without rebuilding sparsity-dependent data.
"""
function setup_smoother end

"""Update a smoother for new coefficients with the same sparsity pattern."""
function update_smoother! end

"""Apply one or more stationary smoothing steps to the current iterate `x`."""
function smooth! end

smoother_steps(config::AbstractSmoother) = config.steps

Base.size(state::SPAI0State) = (length(state.diagonal), length(state.diagonal))
Base.size(state::Union{ILU0State,DILUState}) = (state.n, state.n)
function Base.size(state::AbstractSmootherState, dimension::Integer)
    if dimension == 1 || dimension == 2
        state.n
    else
        1
    end
end
Base.eltype(state::SPAI0State) = eltype(state.diagonal)
Base.eltype(state::ILU0State) = eltype(state.factors)
Base.eltype(state::DILUState) = eltype(state.values)

function check_smoother_dimensions(x, A::StaticSparsityMatrixCSR, b)
    length(x) == matrix_ncols(A) || throw(DimensionMismatch("solution length does not match matrix columns"))
    length(b) == matrix_nrows(A) || throw(DimensionMismatch("right-hand side length does not match matrix rows"))
    matrix_nrows(A) == matrix_ncols(A) || throw(DimensionMismatch("ILU smoothers require a square matrix"))
    nothing
end

function same_smoother_pattern(state, A::StaticSparsityMatrixCSR)
    matrix_nrows(A) == state.n || return false
    matrix_ncols(A) == state.n || return false
    matrix_nonzeros(A) == length(state.host_colval) || return false
    rowptr = host_prefix(A.rowptr, matrix_nrows(A) + 1)
    colval = host_prefix(A.colval, matrix_nonzeros(A))
    rowptr == state.host_rowptr && colval == state.host_colval
end

function require_same_smoother_pattern(state, A::StaticSparsityMatrixCSR)
    same_backend(matrix_backend(A), state.backend) ||
        throw(ArgumentError("smoother and matrix must use the same backend"))
    same_storage = A.rowptr === state.rowptr && A.colval === state.colval &&
                   matrix_nonzeros(A) == length(state.host_colval)
    if !same_storage && !same_smoother_pattern(state, A)
        throw(ArgumentError("smoother update requires an unchanged CSR sparsity pattern"))
    end
    nothing
end

function smoother_residual!(state, A::StaticSparsityMatrixCSR, x, b)
    ensure_smoother_work!(state, b)
    residual!(state.residual, A, x, b)
end

@kernel function axpy_kernel!(x, @Const(y), alpha, n)
    i = @index(Global)
    if i <= n
        @inbounds x[i] += alpha * y[i]
    end
end

function axpy!(x, y, alpha, backend, block_size)
    kernel! = axpy_kernel!(backend, block_size)
    kernel!(x, y, alpha, length(x); ndrange=length(x))
    x
end

function ensure_smoother_work!(state, prototype)
    if isnothing(state.work)
        state.work = similar(prototype)
    end
    if isnothing(state.residual)
        state.residual = similar(prototype)
    end
    state
end

function ensure_smoother_work!(state::SPAI0State, prototype)
    if isnothing(state.temporary)
        state.temporary = similar(prototype)
    end
    if isnothing(state.residual)
        state.residual = similar(prototype)
    end
    state
end

function smooth!(x::AbstractVector, state::AbstractSmootherState,
                 A::StaticSparsityMatrixCSR, b::AbstractVector;
                 steps::Integer=smoother_steps(state.config),
                 zero_initial::Bool=false)
    steps > 0 || throw(ArgumentError("smoothing steps must be positive"))
    check_smoother_dimensions(x, A, b)
    same_backend(matrix_backend(A), state.backend) ||
        throw(ArgumentError("smoother and matrix must use the same backend"))
    if zero_initial
        fill_backend!(x, zero(eltype(x)), matrix_backend(A), matrix_block_size(A))
    end
    for _ in 1:Int(steps)
        smoother_residual!(state, A, x, b)
        apply_correction!(x, state, state.residual)
    end
    x
end

function LinearAlgebra.ldiv!(x::AbstractVector, state::AbstractSmootherState,
                             b::AbstractVector)
    apply!(x, state, b)
end

function LinearAlgebra.mul!(x::AbstractVector, state::AbstractSmootherState,
                            b::AbstractVector)
    apply!(x, state, b)
end

function Base.:\(state::AbstractSmootherState, b::AbstractVector)
    x = similar(b)
    apply!(x, state, b)
end

function Base.:*(state::AbstractSmootherState, b::AbstractVector)
    x = similar(b)
    apply!(x, state, b)
end
