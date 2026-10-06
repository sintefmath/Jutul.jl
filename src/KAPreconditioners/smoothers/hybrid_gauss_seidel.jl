# Correction form of hypre's host relaxation type 6. For each partition B,
# freeze the initial iterate and let r=b-A*x0. The forward and backward
# correction sweeps are
#   (D+w*L_B)*f = w*omega*r,
#   (D+w*U_B)*c = (1-w*omega)*D*f + w*omega*r - w*L_B*f.
# Cross-partition coefficients enter r once and stay frozen through BOTH
# sweeps. Separate f and c buffers avoid races on nonsymmetric sparsity.

# Optional backend-native triangular solves. Their analysis belongs to the
# fixed sparsity pattern; numeric updates retain it and only refresh values.
build_hybrid_native(state, backend) = nothing
update_hybrid_native!(::Nothing, state) = nothing
hybrid_native_storage_bytes(::Nothing) = 0
function solve_hybrid_native! end

function hybrid_partition_count(config, backend, n)
    requested = config.partitions
    if iszero(requested)
        requested = backend isa KernelAbstractions.CPU ? Threads.nthreads() : 1
    end
    return min(requested, n)
end

function hybrid_partitions(n, count, ::Type{Ti}) where {Ti}
    offsets = Int[1]
    ids = Vector{Ti}(undef, n)
    length_per_partition, remainder = divrem(n, count)
    for p in 1:count
        last = offsets[end] + length_per_partition + (p <= remainder)
        fill!(@view(ids[offsets[end]:(last - 1)]), Ti(p))
        push!(offsets, last)
    end
    return offsets, ids
end

@kernel function hybrid_diagonal_kernel!(inverse_diagonal,
        @Const(rp), @Const(cv), @Const(av), n)
    i = @index(Global)
    if i <= n
        diagonal = zero(eltype(av))
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            cv[k] == i && (diagonal += av[k])
        end
        @inbounds inverse_diagonal[i] = iszero(diagonal) ? zero(diagonal) : inv(diagonal)
    end
end

@kernel function hybrid_forward_kernel!(forward, @Const(rhs), @Const(inverse_diagonal),
        @Const(rp), @Const(cv), @Const(av), @Const(ids), @Const(rows), first, count, w, omega)
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = omega * rhs[i]
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            j = cv[k]
            j < i && ids[j] == ids[i] && (value -= av[k] * forward[j])
        end
        @inbounds forward[i] = w * inverse_diagonal[i] * value
    end
end

@kernel function hybrid_backward_kernel!(correction, @Const(forward), @Const(rhs),
        @Const(inverse_diagonal), @Const(rp), @Const(cv), @Const(av),
        @Const(ids), @Const(rows), first, count, w, omega)
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = omega * rhs[i]
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            j = cv[k]
            if ids[j] == ids[i]
                if j < i
                    value -= av[k] * forward[j]
                elseif j > i
                    value -= av[k] * correction[j]
                end
            end
        end
        @inbounds correction[i] = (one(w) - w*omega) * forward[i] + w * inverse_diagonal[i] * value
    end
end

hybrid_outer_weight(state) = convert(matrix_real_type(eltype(state.inverse_diagonal)), state.config.omega)

function setup_hybrid_kernels(state)
    A, backend, block_size = state.matrix, state.backend, state.block_size
    first = state.lower_offsets[1]
    count = state.lower_offsets[2] - first
    w, omega = smoother_damping(state), hybrid_outer_weight(state)
    lower = setup_smoother_kernel(hybrid_forward_kernel!, backend, block_size,
        state.forward, state.residual, state.inverse_diagonal, A.rowptr, A.colval, A.nzval,
        state.partition_ids, state.lower_rows, first, count, w, omega;
        ndrange = count, number_of_launches = length(state.lower_offsets) - 1)
    first = state.upper_offsets[1]
    count = state.upper_offsets[2] - first
    upper = setup_smoother_kernel(hybrid_backward_kernel!, backend, block_size,
        state.work, state.forward, state.residual, state.inverse_diagonal,
        A.rowptr, A.colval, A.nzval, state.partition_ids, state.upper_rows, first, count, w, omega;
        ndrange = count, number_of_launches = length(state.upper_offsets) - 1)
    return lower, upper
end

function setup_smoother(A::StaticSparsityMatrixCSR{Tv, Ti}, config::HybridGaussSeidel;
        reuse = nothing, reallocation_tracker = nothing) where {Tv, Ti}
    require_square_matrix(A, "Hybrid Gauss-Seidel")
    Tv <: Number || throw(ArgumentError("Hybrid Gauss-Seidel requires a scalar matrix"))
    n = matrix_nrows(A)
    n > 0 || throw(ArgumentError("Hybrid Gauss-Seidel requires a nonempty matrix"))
    backend = matrix_backend(A)
    count = hybrid_partition_count(config, backend, n)
    compatible = reuse isa HybridGaussSeidelState &&
        eltype(reuse.inverse_diagonal) === Tv && same_backend(reuse.backend, backend) &&
        length(reuse.partition_offsets) == count + 1 && same_smoother_pattern(reuse, A)
    if compatible
        reuse.config = config
        return update_smoother!(reuse, A)
    end
    host_rp, host_cv = host_prefix(A.rowptr, n + 1), host_prefix(A.colval, matrix_nonzeros(A))
    partition_offsets, ids = hybrid_partitions(n, count, Ti)
    lower_offsets, lower_rows = level_schedule(host_rp, host_cv, n; partition_ids = ids)
    upper_offsets, upper_rows = level_schedule(host_rp, host_cv, n; upper = true, partition_ids = ids)
    mark_backend_reallocation!(reallocation_tracker,
        hybrid_native_storage_bytes(optional_property(reuse, :native)))
    copy_array(key, source) = copy_reusing(optional_property(reuse, key), source, backend;
        reallocation_tracker = reallocation_tracker)
    zeros_array(key) = zeros_reusing(optional_property(reuse, key), backend, Tv, n;
        reallocation_tracker = reallocation_tracker)
    state = HybridGaussSeidelState(A, zeros_array(:inverse_diagonal), A.rowptr, A.colval,
        host_rp, host_cv, copy_array(:partition_ids, ids), partition_offsets,
        lower_offsets, copy_array(:lower_rows, lower_rows),
        upper_offsets, copy_array(:upper_rows, upper_rows),
        zeros_array(:forward), zeros_array(:work), zeros_array(:residual), nothing, nothing,
        config, backend, matrix_kernel_block_size(A), n)
    update_smoother!(state, A)
    state.native = build_hybrid_native(state, backend)
    state.kernels = isnothing(state.native) ? setup_hybrid_kernels(state) : nothing
    return state
end

function update_smoother!(state::HybridGaussSeidelState, A::StaticSparsityMatrixCSR)
    require_same_smoother_pattern(state, A)
    hybrid_partition_count(state.config, state.backend, state.n) == length(state.partition_offsets) - 1 ||
        throw(ArgumentError("Changing Hybrid Gauss-Seidel partitions requires setup_smoother"))
    eltype(A.nzval) === eltype(state.inverse_diagonal) || throw(ArgumentError(
        "Hybrid Gauss-Seidel update requires the same matrix element type"))
    state.matrix, state.rowptr, state.colval = A, A.rowptr, A.colval
    hybrid_diagonal_kernel!(state.backend, state.block_size)(state.inverse_diagonal,
        A.rowptr, A.colval, A.nzval, state.n; ndrange = state.n)
    update_hybrid_native!(state.native, state)
    return state
end

function ensure_smoother_work!(state::HybridGaussSeidelState, prototype)
    if eltype(state.work) !== eltype(prototype)
        state.forward, state.work, state.residual = similar(prototype), similar(prototype), similar(prototype)
        state.native = build_hybrid_native(state, state.backend)
        state.kernels = isnothing(state.native) ? setup_hybrid_kernels(state) : nothing
    end
    return state
end

function hybrid_correction!(state, rhs)
    if !isnothing(state.native)
        return solve_hybrid_native!(state.native, state, rhs)
    end
    A = state.matrix
    w, omega = smoother_damping(state), hybrid_outer_weight(state)
    lower, upper = state.kernels
    for level in 1:(length(state.lower_offsets) - 1)
        first = state.lower_offsets[level]
        count = state.lower_offsets[level + 1] - first
        launch_smoother_kernel(lower, state.forward, rhs, state.inverse_diagonal,
            A.rowptr, A.colval, A.nzval, state.partition_ids, state.lower_rows, first, count, w, omega;
            ndrange = count, launch_index = level)
    end
    for level in 1:(length(state.upper_offsets) - 1)
        first = state.upper_offsets[level]
        count = state.upper_offsets[level + 1] - first
        launch_smoother_kernel(upper, state.work, state.forward, rhs, state.inverse_diagonal,
            A.rowptr, A.colval, A.nzval, state.partition_ids, state.upper_rows, first, count, w, omega;
            ndrange = count, launch_index = level)
    end
    return state.work
end

function hybrid_correction!(state::HybridGaussSeidelState{D, RP}, rhs) where {D, RP <: Vector}
    return hybrid_correction_cpu!(rhs, state.matrix, state.forward, state.work,
        state.inverse_diagonal, state.partition_offsets,
        smoother_damping(state), hybrid_outer_weight(state))
end

function hybrid_correction_cpu!(rhs, A, forward, correction, inverse_diagonal, offsets, w, omega)
    @batch for p in 1:(length(offsets) - 1)
        first, last = offsets[p], offsets[p + 1] - 1
        @inbounds for i in first:last
            value = omega * rhs[i]
            for k in A.rowptr[i]:(A.rowptr[i + 1] - 1)
                j = A.colval[k]
                first <= j < i && (value -= A.nzval[k] * forward[j])
            end
            forward[i] = w * inverse_diagonal[i] * value
        end
        @inbounds for i in last:-1:first
            value = omega * rhs[i]
            for k in A.rowptr[i]:(A.rowptr[i + 1] - 1)
                j = A.colval[k]
                if first <= j < i
                    value -= A.nzval[k] * forward[j]
                elseif i < j <= last
                    value -= A.nzval[k] * correction[j]
                end
            end
            correction[i] = (one(w) - w*omega) * forward[i] + w * inverse_diagonal[i] * value
        end
    end
    return correction
end

solve_hybrid_native_to!(x, factor, state, rhs, add) = false

function hybrid_correction_to!(x, state::HybridGaussSeidelState, rhs, add::Bool)
    ensure_smoother_work!(state, rhs)
    if !isnothing(state.native) && solve_hybrid_native_to!(x, state.native, state, rhs, add)
        return x
    end
    correction = hybrid_correction!(state, rhs)
    if add
        axpy!(x, correction, one(smoother_damping(state)), state.backend, state.block_size)
    else
        copyto!(x, correction)
    end
    return x
end

apply_correction!(x, state::HybridGaussSeidelState, rhs) = hybrid_correction_to!(x, state, rhs, true)

function smooth_level!(x, A::StaticSparsityMatrixCSR, b, state::HybridGaussSeidelState,
        steps::Int; residual = nothing, zero_initial::Bool = false)
    steps > 0 || throw(ArgumentError("smoothing steps must be positive"))
    check_smoother_dimensions(x, A, b)
    require_smoother_size_and_backend(state, A, "Hybrid Gauss-Seidel")
    ensure_smoother_work!(state, b)
    start = 1
    if zero_initial
        rhs = isnothing(residual) ? b : residual
        hybrid_correction_to!(x, state, rhs, false)
        start = 2
    elseif !isnothing(residual)
        apply_correction!(x, state, residual)
        start = 2
    end
    for _ in start:steps
        residual!(state.residual, A, x, b)
        apply_correction!(x, state, state.residual)
    end
    return x
end

smooth_result!(x, A::StaticSparsityMatrixCSR, b, state::HybridGaussSeidelState, steps::Int;
        kwargs...) = smooth_level!(x, A, b, state, steps; kwargs...)

smooth!(x::AbstractVector, state::HybridGaussSeidelState, A::StaticSparsityMatrixCSR,
        b::AbstractVector; steps::Integer = state.config.steps, zero_initial::Bool = false) =
    smooth_level!(x, A, b, state, Int(steps); zero_initial = zero_initial)

function apply!(x::AbstractVector, state::HybridGaussSeidelState, b::AbstractVector)
    require_apply_dimensions(x, state, b)
    same_backend(KernelAbstractions.get_backend(x), state.backend) ||
        throw(ArgumentError("output and smoother must use the same backend"))
    return smooth_level!(x, state.matrix, b, state, state.config.steps; zero_initial = true)
end
