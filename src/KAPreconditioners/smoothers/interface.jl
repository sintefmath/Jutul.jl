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

setup_smoother(
    A::SparseMatrixCSC,
    config::Union{SPAI0, GaussSeidel, ILU0, DILU, VendorILU} = SPAI0(); kwargs...
) =
    setup_smoother(csr_matrix(A), config; kwargs...)

smoother_steps(config::AbstractSmoother) = config.steps

Base.size(state::SPAI0State) = (length(state.diagonal), length(state.diagonal))
Base.size(state::GaussSeidelState) = (state.n, state.n)
Base.size(state::Union{ILU0State, DILUState, VendorILUState}) = (state.n, state.n)
function Base.size(state::AbstractSmootherState, dimension::Integer)
    return if dimension == 1 || dimension == 2
        state.n
    else
        1
    end
end
Base.eltype(state::SPAI0State) = eltype(state.diagonal)
Base.eltype(state::GaussSeidelState) = eltype(state.inverse_diagonal)
Base.eltype(state::ILU0State) = eltype(state.factors)
Base.eltype(state::DILUState) = eltype(state.values)
Base.eltype(::VendorILUState{Tv}) where {Tv} = Tv

function smoother_damping(state::SPAI0State)
    T = matrix_real_type(eltype(state.diagonal))
    return convert(T, state.config.damping)
end

function smoother_damping(state::Union{GaussSeidelState, ILU0State, DILUState})
    T = matrix_real_type(eltype(state.inverse_diagonal))
    return convert(T, state.config.damping)
end

function smoother_damping(state::VendorILUState)
    T = matrix_real_type(eltype(state.factor_values))
    return convert(T, state.config.damping)
end

function check_smoother_dimensions(x, A::StaticSparsityMatrixCSR, b)
    length(x) == matrix_ncols(A) || throw(DimensionMismatch("solution length does not match matrix columns"))
    length(b) == matrix_nrows(A) || throw(DimensionMismatch("right-hand side length does not match matrix rows"))
    matrix_nrows(A) == matrix_ncols(A) || throw(DimensionMismatch("ILU smoothers require a square matrix"))
    return nothing
end

function require_square_matrix(
        A::StaticSparsityMatrixCSR,
        name::AbstractString
    )
    matrix_nrows(A) == matrix_ncols(A) || throw(
        DimensionMismatch(
            "$name requires a square matrix"
        )
    )
    return nothing
end

function same_smoother_pattern(state, A::StaticSparsityMatrixCSR)
    matrix_nrows(A) == state.n || return false
    matrix_ncols(A) == state.n || return false
    matrix_nonzeros(A) == length(state.host_colval) || return false
    if A.rowptr === state.rowptr && A.colval === state.colval
        return true
    end
    rowptr = host_prefix(A.rowptr, matrix_nrows(A) + 1)
    colval = host_prefix(A.colval, matrix_nonzeros(A))
    return rowptr == state.host_rowptr && colval == state.host_colval
end

smoother_state_size(state) = state.n
smoother_state_size(state::SPAI0State) = length(state.diagonal)

function require_smoother_size(
        state, A::StaticSparsityMatrixCSR,
        name::AbstractString
    )
    smoother_state_size(state) == matrix_nrows(A) || throw(
        ArgumentError(
            "$name state size does not match the matrix"
        )
    )
    return nothing
end

function require_smoother_size_and_backend(
        state, A::StaticSparsityMatrixCSR,
        name::AbstractString
    )
    require_smoother_size(state, A, name)
    same_backend(matrix_backend(A), state.backend) || throw(
        ArgumentError(
            "smoother and matrix must use the same backend"
        )
    )
    return nothing
end

function require_apply_dimensions(x, state, b)
    n = smoother_state_size(state)
    length(x) == n || throw(DimensionMismatch())
    length(b) == n || throw(DimensionMismatch())
    return nothing
end

function require_same_smoother_pattern(state, A::StaticSparsityMatrixCSR)
    same_backend(matrix_backend(A), state.backend) ||
        throw(ArgumentError("smoother and matrix must use the same backend"))
    same_storage = A.rowptr === state.rowptr && A.colval === state.colval &&
        matrix_nonzeros(A) == length(state.host_colval)
    if !same_storage && !same_smoother_pattern(state, A)
        throw(ArgumentError("smoother update requires an unchanged CSR sparsity pattern"))
    end
    return nothing
end

function smoother_residual!(state, A::StaticSparsityMatrixCSR, x, b)
    ensure_smoother_work!(state, b)
    return residual!(state.residual, A, x, b)
end

@kernel function axpy_kernel!(x, @Const(y), alpha, n)
    i = @index(Global)
    if i <= n
        @inbounds x[i] += alpha * y[i]
    end
end

function axpy!(x, y, alpha, backend, block_size)
    kernel! = axpy_kernel!(backend, block_size)
    kernel!(x, y, alpha, length(x); ndrange = length(x))
    return x
end

function axpy!(
        x::Vector, y::Vector, alpha,
        ::KernelAbstractions.CPU, min_batch
    )
    foreach_cpu_row(length(x), min_batch) do i
        @inbounds x[i] += alpha * y[i]
    end
    return x
end

# Backends can use the representative arguments to compile a reusable launch
# object. The default keeps the ordinary KernelAbstractions kernel.
function setup_smoother_kernel(
        kernel, backend, block_size, arguments...;
        ndrange, number_of_launches = 1
    )
    return kernel(backend, block_size)
end

function launch_smoother_kernel(
        kernel, arguments...; ndrange, launch_index = 1
    )
    kernel(arguments...; ndrange = ndrange)
    return nothing
end

function ensure_smoother_work!(state, prototype)
    state.work = ensure_smoother_buffer(state.work, prototype)
    state.residual = ensure_smoother_buffer(state.residual, prototype)
    return state
end

function ensure_smoother_work!(state::SPAI0State, prototype)
    state.temporary = ensure_smoother_buffer(state.temporary, prototype)
    state.residual = ensure_smoother_buffer(state.residual, prototype)
    return state
end

ensure_smoother_buffer(buffer, prototype) =
    isnothing(buffer) ? similar(prototype) : buffer

function update_level_smoother!(
        state::Union{
            SPAI0State, GaussSeidelState, ILU0State, DILUState, VendorILUState,
        },
        A::StaticSparsityMatrixCSR, options::AMGOptions
    )
    state.config = options.smoother
    return update_smoother!(state, A)
end

function copy_result!(destination, result)
    result === destination || copyto!(destination, result)
    return destination
end

function smooth!(
        x::AbstractVector, state::AbstractSmootherState,
        A::StaticSparsityMatrixCSR, b::AbstractVector;
        steps::Integer = smoother_steps(state.config),
        zero_initial::Bool = false
    )
    steps > 0 || throw(ArgumentError("smoothing steps must be positive"))
    check_smoother_dimensions(x, A, b)
    same_backend(matrix_backend(A), state.backend) ||
        throw(ArgumentError("smoother and matrix must use the same backend"))
    if zero_initial
        fill_backend!(x, zero(eltype(x)), matrix_backend(A), matrix_batch_size(A))
    end
    for _ in 1:Int(steps)
        smoother_residual!(state, A, x, b)
        apply_correction!(x, state, state.residual)
    end
    return x
end

function LinearAlgebra.ldiv!(
        x::AbstractVector, state::AbstractSmootherState,
        b::AbstractVector
    )
    return apply!(x, state, b)
end

function LinearAlgebra.mul!(
        x::AbstractVector, state::AbstractSmootherState,
        b::AbstractVector
    )
    return apply!(x, state, b)
end

function Base.:\(state::AbstractSmootherState, b::AbstractVector)
    x = similar(b)
    return apply!(x, state, b)
end

function Base.:*(state::AbstractSmootherState, b::AbstractVector)
    x = similar(b)
    return apply!(x, state, b)
end
