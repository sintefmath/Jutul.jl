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

function setup_smoother(A::StaticSparsityMatrixCSR{Tv}, config::SPAI0=SPAI0();
        reuse=nothing, reallocation_tracker=nothing) where Tv
    require_square_matrix(A, "SPAI0")
    compatible = reuse isa SPAI0State && reuse.n == matrix_nrows(A) &&
                 same_backend(reuse.backend, matrix_backend(A)) &&
                 eltype(reuse.diagonal) === Tv
    if compatible
        reuse.config = config
        return update_smoother!(reuse, A)
    end
    old_diagonal = optional_property(reuse, :diagonal)
    old_temporary = optional_property(reuse, :temporary)
    old_residual = optional_property(reuse, :residual)
    backend = matrix_backend(A)
    n = matrix_nrows(A)
    diagonal = zeros_reusing(old_diagonal, backend, Tv,
        n; reallocation_tracker=reallocation_tracker)
    if Tv <: Number
        temporary = zeros_reusing(old_temporary, backend, Tv, n;
            reallocation_tracker=reallocation_tracker)
        residual = zeros_reusing(old_residual, backend, Tv, n;
            reallocation_tracker=reallocation_tracker)
    else
        temporary = nothing
        residual = nothing
    end
    state = SPAI0State(diagonal, temporary, residual, config,
                       backend, matrix_batch_size(A), n)
    update_smoother!(state, A)
end

function update_smoother!(state::SPAI0State, A::StaticSparsityMatrixCSR)
    require_smoother_size_and_backend(state, A, "SPAI0")
    n = matrix_nrows(A)
    kernel! = spai0_setup_kernel!(
        matrix_backend(A), matrix_kernel_block_size(A))
    kernel!(state.diagonal, A.rowptr, A.colval, A.nzval, smoother_damping(state),
            n; ndrange=n)
    state
end

function update_smoother!(state::SPAI0State{D},
        A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector}) where {D<:Vector,Tv,Ti}
    require_smoother_size_and_backend(state, A, "SPAI0")
    diagonal = state.diagonal
    damping = smoother_damping(state)
    foreach_cpu_row(matrix_nrows(A), matrix_batch_size(A)) do i
        aii = zero(eltype(diagonal))
        scale = zero(damping)
        @inbounds for k in A.rowptr[i]:(A.rowptr[i+1]-1)
            value = A.nzval[k]
            scale += spai0_scale(value, damping)
            A.colval[k] == i && (aii += value)
        end
        @inbounds diagonal[i] = spai0_inverse(aii, scale, damping)
    end
    state
end

function apply!(x::Vector, state::SPAI0State{D}, b::Vector) where {D<:Vector}
    require_apply_dimensions(x, state, b)
    diagonal = state.diagonal
    foreach_cpu_row(length(x), state.block_size) do i
        @inbounds x[i] = diagonal[i] * b[i]
    end
    x
end

function apply_correction!(x::Vector, state::SPAI0State{D},
        residual::Vector) where {D<:Vector}
    diagonal = state.diagonal
    foreach_cpu_row(length(x), state.block_size) do i
        @inbounds x[i] += diagonal[i] * residual[i]
    end
    x
end

function apply!(x::AbstractVector, state::SPAI0State, b::AbstractVector)
    require_apply_dimensions(x, state, b)
    ensure_smoother_work!(state, b)
    backend = state.backend
    n = length(x)
    same_backend(KernelAbstractions.get_backend(x), backend) ||
        throw(ArgumentError("output and smoother must use the same backend"))
    kernel! = spai0_apply_kernel!(backend, state.block_size)
    kernel!(x, b, state.diagonal, n; ndrange=n)
    x
end

function apply_correction!(x, state::SPAI0State, residual)
    backend = state.backend
    kernel! = spai0_apply_kernel!(backend, state.block_size)
    kernel!(state.temporary, residual, state.diagonal, length(x); ndrange=length(x))
    axpy!(x, state.temporary, one(smoother_damping(state)), backend,
           state.block_size)
    x
end


function smooth_result!(x, A::StaticSparsityMatrixCSR, b, state::SPAI0State, steps::Int;
                         residual=nothing, zero_initial::Bool=false)
    ensure_smoother_work!(state, b)
    backend = matrix_backend(A)
    kernel_block_size = matrix_kernel_block_size(A)
    n = matrix_nrows(A)
    step! = spai0_step_kernel!(backend, kernel_block_size)
    residual_step! = spai0_residual_step_kernel!(backend, kernel_block_size)
    zero_step! = spai0_zero_residual_step_kernel!(backend, kernel_block_size)
    start = 1
    if !isnothing(residual)
        if zero_initial
            zero_step!(x, residual, state.diagonal, n; ndrange=n)
        else
            residual_step!(x, x, residual, state.diagonal, n; ndrange=n)
        end
        steps == 1 && return x
        start = 2
    end
    src, dst = x, state.temporary
    for _ in start:steps
        step!(dst, src, b, state.diagonal, A.rowptr, A.colval, A.nzval,
              n; ndrange=n)
        src, dst = dst, src
    end
    src
end

function spai0_step_cpu!(dst, src, A, b, diagonal)
    foreach_cpu_row(matrix_nrows(A), matrix_batch_size(A)) do i
        value = b[i]
        @inbounds @simd for k in A.rowptr[i]:(A.rowptr[i+1]-1)
            value -= A.nzval[k] * src[A.colval[k]]
        end
        @inbounds dst[i] = src[i] + diagonal[i] * value
    end
    dst
end

function smooth_result!(x::Vector,
        A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector},
        b::Vector, state::SPAI0State{D}, steps::Int;
        residual=nothing, zero_initial::Bool=false) where {Tv,Ti,D<:Vector}
    start = 1
    diagonal = state.diagonal
    if !isnothing(residual)
        if zero_initial
            foreach_cpu_row(matrix_nrows(A), state.block_size) do i
                @inbounds x[i] = diagonal[i] * residual[i]
            end
        else
            foreach_cpu_row(matrix_nrows(A), state.block_size) do i
                @inbounds x[i] += diagonal[i] * residual[i]
            end
        end
        steps == 1 && return x
        start = 2
    end
    src, dst = x, state.temporary
    for _ in start:steps
        spai0_step_cpu!(dst, src, A, b, diagonal)
        src, dst = dst, src
    end
    src
end

function smooth_level!(x, A::StaticSparsityMatrixCSR, b, state::SPAI0State, steps::Int;
                  residual=nothing, zero_initial::Bool=false)
    result = smooth_result!(x, A, b, state, steps;
                             residual=residual, zero_initial=zero_initial)
    copy_result!(x, result)
end

function smooth_once_to!(dst, src, A::StaticSparsityMatrixCSR, b, state::SPAI0State)
    n = matrix_nrows(A)
    kernel! = spai0_step_kernel!(
        matrix_backend(A), matrix_kernel_block_size(A))
    kernel!(dst, src, b, state.diagonal, A.rowptr, A.colval, A.nzval,
            n; ndrange=n)
    dst
end

function smooth_once_to!(dst::Vector, src::Vector,
        A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector},
        b::Vector, state::SPAI0State{D}) where {Tv,Ti,D<:Vector}
    spai0_step_cpu!(dst, src, A, b, state.diagonal)
end

function zero_smooth_residual!(x, residual, A::StaticSparsityMatrixCSR, b,
                                state::SPAI0State)
    n = matrix_nrows(A)
    kernel! = zero_spai0_residual_kernel!(
        matrix_backend(A), matrix_kernel_block_size(A))
    kernel!(x, residual, b, state.diagonal, A.rowptr, A.colval, A.nzval,
            n; ndrange=n)
    residual
end


function zero_smooth_residual!(x::Vector, residual::Vector,
        A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector},
        b::Vector, state::SPAI0State{D}) where {Tv,Ti,D<:Vector}
    diagonal = state.diagonal
    foreach_cpu_row(matrix_nrows(A), state.block_size) do i
        ax = zero(eltype(residual))
        @inbounds @simd for k in A.rowptr[i]:(A.rowptr[i+1]-1)
            j = A.colval[k]
            ax += A.nzval[k] * diagonal[j] * b[j]
        end
        @inbounds begin
            x[i] = diagonal[i] * b[i]
            residual[i] = b[i] - ax
        end
    end
    residual
end
