@inline spai0_scale(value::Number, damping) = abs2(value)
@inline spai0_scale(value, damping) = zero(damping)

@inline function spai0_inverse(diagonal::Number, scale, damping)
    if iszero(scale)
        zero(diagonal)
    else
        damping * diagonal / scale
    end
end

@inline spai0_inverse(diagonal, scale, damping) = damping * inv(diagonal)

@kernel function spai0_setup_kernel!(d, @Const(rp), @Const(cv), @Const(av), damping, n)
    i = @index(Global)
    if i <= n
        diagonal = zero(eltype(d))
        scale = zero(damping)
        @inbounds for k in rp[i]:(rp[i+1]-1)
            value = av[k]
            scale += spai0_scale(value, damping)
            cv[k] == i && (diagonal += value)
        end
        @inbounds d[i] = spai0_inverse(diagonal, scale, damping)
    end
end

@kernel function spai0_step_kernel!(dst, @Const(src), @Const(b), @Const(d),
                                     @Const(rp), @Const(cv), @Const(av), n)
    i = @index(Global)
    if i <= n
        residual = b[i]
        @inbounds for k in rp[i]:(rp[i+1]-1)
            residual -= av[k] * src[cv[k]]
        end
        @inbounds dst[i] = src[i] + d[i] * residual
    end
end

@kernel function spai0_apply_kernel!(x, @Const(b), @Const(d), n)
    i = @index(Global)
    if i <= n
        @inbounds x[i] = d[i] * b[i]
    end
end

@kernel function spai0_residual_step_kernel!(dst, @Const(src), @Const(residual),
                                              @Const(d), n)
    i = @index(Global)
    if i <= n
        @inbounds dst[i] = src[i] + d[i] * residual[i]
    end
end

@kernel function spai0_zero_residual_step_kernel!(dst, @Const(residual),
                                                   @Const(d), n)
    i = @index(Global)
    if i <= n
        @inbounds dst[i] = d[i] * residual[i]
    end
end

@kernel function zero_spai0_residual_kernel!(x, residual, @Const(b), @Const(d),
                                              @Const(rp), @Const(cv),
                                              @Const(av), n)
    i = @index(Global)
    if i <= n
        ax = zero(eltype(residual))
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            j = cv[k]
            ax += av[k] * d[j] * b[j]
        end
        @inbounds begin
            x[i] = d[i] * b[i]
            residual[i] = b[i] - ax
        end
    end
end

function setup_smoother(A::StaticSparsityMatrixCSR{Tv}, config::SPAI0=SPAI0(); reuse=nothing) where Tv
    matrix_nrows(A) == matrix_ncols(A) || throw(DimensionMismatch("SPAI0 requires a square matrix"))
    compatible = reuse isa SPAI0State && reuse.n == matrix_nrows(A) &&
                 reuse.backend === matrix_backend(A) && eltype(reuse.diagonal) === Tv
    if compatible
        reuse.config = config
        return update_smoother!(reuse, A)
    end
    if reuse isa SPAI0State
        old_diagonal = reuse.diagonal
        old_temporary = reuse.temporary
        old_residual = reuse.residual
    else
        old_diagonal = nothing
        old_temporary = nothing
        old_residual = nothing
    end
    diagonal = zeros_reusing(old_diagonal, matrix_backend(A), Tv, matrix_nrows(A))
    if Tv <: Number
        temporary = zeros_reusing(old_temporary, matrix_backend(A), Tv, matrix_nrows(A))
        residual = zeros_reusing(old_residual, matrix_backend(A), Tv, matrix_nrows(A))
    else
        temporary = nothing
        residual = nothing
    end
    state = SPAI0State(diagonal, temporary, residual, config,
                       matrix_backend(A), matrix_block_size(A), matrix_nrows(A))
    update_smoother!(state, A)
end

setup_smoother(A::SparseMatrixCSC, config::SPAI0=SPAI0(); reuse=nothing) =
    setup_smoother(csr_matrix(A), config; reuse=reuse)

function update_smoother!(state::SPAI0State, A::StaticSparsityMatrixCSR)
    length(state.diagonal) == matrix_nrows(A) ||
        throw(ArgumentError("SPAI0 state size does not match the matrix"))
    matrix_backend(A) === state.backend ||
        throw(ArgumentError("smoother and matrix must use the same backend"))
    kernel! = spai0_setup_kernel!(matrix_backend(A), matrix_block_size(A))
    kernel!(state.diagonal, A.rowptr, A.colval, A.nzval, state.config.damping,
            matrix_nrows(A); ndrange=matrix_nrows(A))
    state
end

function apply!(x::AbstractVector, state::SPAI0State, b::AbstractVector)
    length(x) == length(state.diagonal) || throw(DimensionMismatch())
    length(b) == length(state.diagonal) || throw(DimensionMismatch())
    ensure_smoother_work!(state, b)
    backend = state.backend
    KernelAbstractions.get_backend(x) === backend ||
        throw(ArgumentError("output and smoother must use the same backend"))
    kernel! = spai0_apply_kernel!(backend, state.block_size)
    kernel!(x, b, state.diagonal, length(x); ndrange=length(x))
    x
end

function apply_correction!(x, state::SPAI0State, residual)
    backend = state.backend
    kernel! = spai0_apply_kernel!(backend, state.block_size)
    kernel!(state.temporary, residual, state.diagonal, length(x); ndrange=length(x))
    axpy!(x, state.temporary, one(state.config.damping), backend,
           state.block_size)
    x
end


function smooth_result!(x, A::StaticSparsityMatrixCSR, b, state::SPAI0State, steps::Int;
                         residual=nothing, zero_initial::Bool=false)
    ensure_smoother_work!(state, b)
    step! = spai0_step_kernel!(matrix_backend(A), matrix_block_size(A))
    residual_step! = spai0_residual_step_kernel!(matrix_backend(A), matrix_block_size(A))
    zero_step! = spai0_zero_residual_step_kernel!(matrix_backend(A), matrix_block_size(A))
    start = 1
    if !isnothing(residual)
        if zero_initial
            zero_step!(x, residual, state.diagonal, matrix_nrows(A); ndrange=matrix_nrows(A))
        else
            residual_step!(x, x, residual, state.diagonal, matrix_nrows(A); ndrange=matrix_nrows(A))
        end
        steps == 1 && return x
        start = 2
    end
    src, dst = x, state.temporary
    for _ in start:steps
        step!(dst, src, b, state.diagonal, A.rowptr, A.colval, A.nzval,
              matrix_nrows(A); ndrange=matrix_nrows(A))
        src, dst = dst, src
    end
    src
end

function smooth_level!(x, A::StaticSparsityMatrixCSR, b, state::SPAI0State, steps::Int;
                  residual=nothing, zero_initial::Bool=false)
    result = smooth_result!(x, A, b, state, steps;
                             residual=residual, zero_initial=zero_initial)
    result === x || copyto!(x, result)
    x
end

function smooth_once_to!(dst, src, A::StaticSparsityMatrixCSR, b, state::SPAI0State)
    kernel! = spai0_step_kernel!(matrix_backend(A), matrix_block_size(A))
    kernel!(dst, src, b, state.diagonal, A.rowptr, A.colval, A.nzval,
            matrix_nrows(A); ndrange=matrix_nrows(A))
    dst
end

function zero_smooth_residual!(x, residual, A::StaticSparsityMatrixCSR, b,
                                state::SPAI0State)
    kernel! = zero_spai0_residual_kernel!(matrix_backend(A), matrix_block_size(A))
    kernel!(x, residual, b, state.diagonal, A.rowptr, A.colval, A.nzval,
            matrix_nrows(A); ndrange=matrix_nrows(A))
    residual
end

function update_level_smoother!(state::SPAI0State, A::StaticSparsityMatrixCSR, options::AMGOptions)
    state.config = options.smoother
    update_smoother!(state, A)
end

