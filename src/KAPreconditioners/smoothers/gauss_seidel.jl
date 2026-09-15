function setup_smoother(A::StaticSparsityMatrixCSR{Tv},
        config::GaussSeidel; reuse=nothing) where Tv
    matrix_nrows(A) == matrix_ncols(A) || throw(DimensionMismatch(
        "Gauss-Seidel requires a square matrix"))
    matrix_backend(A) isa KernelAbstractions.CPU || throw(ArgumentError(
        "Gauss-Seidel is only available on the CPU backend"))
    compatible = reuse isa GaussSeidelState && reuse.n == matrix_nrows(A) &&
                 same_backend(reuse.backend, matrix_backend(A)) &&
                 eltype(reuse.inverse_diagonal) === Tv
    if compatible
        reuse.config = config
        return update_smoother!(reuse, A)
    end
    n = matrix_nrows(A)
    inverse_diagonal = Vector{Tv}(undef, n)
    correction = zeros(Tv, n)
    residual = zeros(Tv, n)
    state = GaussSeidelState(A, inverse_diagonal, correction, residual,
        config, matrix_backend(A), matrix_block_size(A), n)
    update_smoother!(state, A)
end

setup_smoother(A::SparseMatrixCSC, config::GaussSeidel;
        reuse=nothing) = setup_smoother(csr_matrix(A), config; reuse=reuse)

function update_smoother!(state::GaussSeidelState,
        A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector}) where {Tv,Ti}
    state.n == matrix_nrows(A) || throw(ArgumentError(
        "Gauss-Seidel state size does not match the matrix"))
    state.matrix = A
    @inbounds for i in 1:state.n
        diagonal = zero(Tv)
        for k in A.rowptr[i]:(A.rowptr[i+1]-1)
            A.colval[k] == i && (diagonal += A.nzval[k])
        end
        iszero(diagonal) && throw(ArgumentError(
            "Gauss-Seidel requires a nonzero diagonal in row $i"))
        # BoomerAMG relaxation types 13/14 use option-4 L1 norms. On one MPI
        # rank there is no off-processor block, so that norm is abs(A_ii).
        state.inverse_diagonal[i] = inv(abs(diagonal))
    end
    state
end

function gauss_seidel_sweep!(x, A, b, inverse_diagonal, damping,
        rows)
    @inbounds for i in rows
        value = b[i]
        @simd for k in A.rowptr[i]:(A.rowptr[i+1]-1)
            j = A.colval[k]
            value -= A.nzval[k] * x[j]
        end
        x[i] += damping * inverse_diagonal[i] * value
    end
    x
end

function gauss_seidel_correction!(x, correction, A, residual,
        inverse_diagonal, damping, rows)
    fill!(correction, zero(eltype(correction)))
    @inbounds for i in rows
        value = residual[i]
        @simd for k in A.rowptr[i]:(A.rowptr[i+1]-1)
            j = A.colval[k]
            value -= A.nzval[k] * correction[j]
        end
        correction[i] = damping * inverse_diagonal[i] * value
        x[i] += correction[i]
    end
    x
end

function smooth_level!(x::Vector,
        A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector},
        b::Vector, state::GaussSeidelState, steps::Int;
        residual=nothing, zero_initial::Bool=false) where {Tv,Ti}
    if zero_initial
        fill!(x, zero(eltype(x)))
    end
    start = 1
    if !isnothing(residual) && !zero_initial
        gauss_seidel_correction!(x, state.correction, A, residual,
            state.inverse_diagonal, state.config.damping, 1:state.n)
        start = 2
    end
    for _ in start:steps
        gauss_seidel_sweep!(x, A, b, state.inverse_diagonal,
            state.config.damping, 1:state.n)
    end
    x
end

function smooth_result!(x::Vector,
        A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector},
        b::Vector, state::GaussSeidelState, steps::Int;
        residual=nothing, zero_initial::Bool=false) where {Tv,Ti}
    if !isnothing(residual) && !zero_initial
        gauss_seidel_correction!(x, state.correction, A, residual,
            state.inverse_diagonal, state.config.damping, state.n:-1:1)
        start = 2
    else
        zero_initial && fill!(x, zero(eltype(x)))
        start = 1
    end
    for _ in start:steps
        gauss_seidel_sweep!(x, A, b, state.inverse_diagonal,
            state.config.damping, state.n:-1:1)
    end
    x
end

function apply!(x::Vector, state::GaussSeidelState, b::Vector)
    # Symmetric application for stand-alone use. AMG uses the direction-aware
    # methods above for its down and up cycles.
    fill!(x, zero(eltype(x)))
    gauss_seidel_sweep!(x, state.matrix, b, state.inverse_diagonal,
        state.config.damping, 1:state.n)
    residual!(state.residual, state.matrix, x, b)
    gauss_seidel_correction!(x, state.correction, state.matrix,
        state.residual, state.inverse_diagonal, state.config.damping,
        state.n:-1:1)
    x
end

function update_level_smoother!(state::GaussSeidelState,
        A::StaticSparsityMatrixCSR, options::AMGOptions)
    state.config = options.smoother
    update_smoother!(state, A)
end
