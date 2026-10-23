function hybrid_partition(A, config, partition)
    n = matrix_nrows(A)
    if isnothing(partition)
        np = min(config.npartitions, max(1, n ÷ config.min_partition_size))
        if np == 1
            return ones(Int, n)
        elseif config.partitioner == :contiguous
            return [1 + (i - 1) * np ÷ n for i in 1:n]
        else
            # Partition the symmetrized sparsity graph, independently of signs
            # and numerical zeros. The schedule is reused after value updates.
            graph = SparseMatrixCSC(n, n, copy(A.rowptr), copy(A.colval), ones(Int, matrix_nonzeros(A)))
            return Int.(Jutul.partition(Jutul.MetisPartitioner(), graph, np))
        end
    end
    length(partition) == n || throw(DimensionMismatch("partition must contain one owner per row"))
    owners = Int.(partition)
    isempty(owners) && return owners
    labels = sort!(unique(owners))
    labels == collect(1:length(labels)) || throw(ArgumentError("partition labels must be consecutive positive integers"))
    return owners
end

function hybrid_schedule(A, owners)
    n = matrix_nrows(A)
    np = maximum(owners; init = 1)
    offsets = ones(Int, np + 1)
    for p in owners
        offsets[p + 1] += 1
    end
    for p in 1:np
        offsets[p + 1] += offsets[p] - 1
    end
    # Keep contiguous ownership as a range: this removes a full-size row map
    # and its dependent load from the hot path on well-ordered meshes.
    if issorted(owners)
        rows = 1:n
    else
        next_row = copy(offsets)
        rows = Vector{Int}(undef, n)
        for i in 1:n
            p = owners[i]
            rows[next_row[p]] = i
            next_row[p] += 1
        end
    end
    # Interior rows use the original CSR directly. Only interface rows need
    # split local/remote entry lists; no per-nonzero ownership test in a sweep.
    boundary_index = zeros(Int, n)
    diagonal_positions = zeros(eltype(A.rowptr), n)
    boundary_offsets = Int[1]
    boundary_rows = Int[]
    local_diagonal_positions = Int[]
    local_offsets, remote_offsets = Int[1], Int[1]
    local_positions, remote_positions = eltype(A.rowptr)[], eltype(A.rowptr)[]
    sorted_rows = true
    for p in 1:np
        for t in offsets[p]:(offsets[p + 1] - 1)
            i = rows[t]
            entries = A.rowptr[i]:(A.rowptr[i + 1] - 1)
            boundary = false
            diagonal = zero(eltype(A.rowptr))
            diagonal_count = 0
            for k in entries
                j = A.colval[k]
                sorted_rows &= k == first(entries) || A.colval[k - 1] < j
                boundary |= owners[j] != p
                if j == i
                    diagonal = k
                    diagonal_count += 1
                end
            end
            # Zero marks missing/duplicate diagonals, including unsigned CSR.
            if diagonal_count == 1
                diagonal_positions[i] = diagonal
            end
            boundary || continue
            push!(boundary_rows, i)
            boundary_index[i] = length(boundary_rows)
            local_diagonal = 0
            for k in entries
                j = A.colval[k]
                if owners[j] == p
                    push!(local_positions, k)
                    if j == i
                        local_diagonal = length(local_positions)
                    end
                else
                    push!(remote_positions, k)
                end
            end
            push!(local_diagonal_positions, local_diagonal)
            push!(local_offsets, length(local_positions) + 1)
            push!(remote_offsets, length(remote_positions) + 1)
        end
        push!(boundary_offsets, length(boundary_rows) + 1)
    end
    return (;
        owners, offsets, rows, diagonal_positions, boundary_index, boundary_offsets,
        boundary_rows, local_offsets, local_positions, remote_offsets, remote_positions,
        local_diagonal_positions,
        # Short rows favor the existing predicated scan. On longer sorted
        # rows, cached triangular ranges skip the unused half entirely.
        triangular_ranges = Val(sorted_rows && matrix_nonzeros(A) >= 12n),
    )
end

@inline function hybrid_triangular_value(x, A, rhs, schedule, i, ::Val{forward}) where {forward}
    value = @inbounds rhs[i]
    h = @inbounds schedule.boundary_index[i]
    if h == 0
        diagonal = @inbounds schedule.diagonal_positions[i]
        if forward
            lo = @inbounds A.rowptr[i]
            hi = diagonal - 1
        else
            lo = diagonal + 1
            hi = @inbounds A.rowptr[i + 1] - 1
        end
        @inbounds @simd for k in lo:hi
            value -= A.nzval[k] * x[A.colval[k]]
        end
    else
        diagonal = @inbounds schedule.local_diagonal_positions[h]
        if forward
            lo = @inbounds schedule.local_offsets[h]
            hi = diagonal - 1
        else
            lo = diagonal + 1
            hi = @inbounds schedule.local_offsets[h + 1] - 1
        end
        @inbounds @simd for t in lo:hi
            k = schedule.local_positions[t]
            value -= A.nzval[k] * x[A.colval[k]]
        end
    end
    return value
end

function setup_smoother(
        A::StaticSparsityMatrixCSR, config::HybridGaussSeidel;
        reuse = nothing, partition = nothing, reallocation_tracker = nothing
    )
    require_square_matrix(A, "Hybrid Gauss-Seidel")
    cpu_storage = matrix_backend(A) isa KernelAbstractions.CPU &&
        A.rowptr isa Vector && A.colval isa Vector && A.nzval isa Vector
    if !cpu_storage
        throw(ArgumentError("Hybrid Gauss-Seidel requires CPU Vector CSR storage"))
    end
    T = eltype(A.nzval)
    T <: AbstractFloat || throw(ArgumentError("Hybrid Gauss-Seidel requires real floating-point scalar coefficients"))
    compatible = reuse isa HybridGaussSeidelState && eltype(reuse) === T &&
        eltype(reuse.rowptr) === eltype(A.rowptr) &&
        same_smoother_pattern(reuse, A) &&
        reuse.config.npartitions == config.npartitions &&
        reuse.config.min_partition_size == config.min_partition_size &&
        reuse.config.partitioner == config.partitioner &&
        (isnothing(partition) || reuse.schedule.owners == partition)
    if compatible
        reuse.config = config
        return update_smoother!(reuse, A)
    end
    schedule = hybrid_schedule(A, hybrid_partition(A, config, partition))
    n = matrix_nrows(A)
    state = HybridGaussSeidelState(
        A, Vector{T}(undef, n), Vector{T}(undef, n),
        config, matrix_backend(A), matrix_batch_size(A), n, A.rowptr, A.colval,
        copy(A.rowptr), copy(A.colval), schedule, Vector{T}(undef, length(schedule.boundary_rows))
    )
    return update_smoother!(state, A)
end

function update_smoother!(
        state::HybridGaussSeidelState,
        A::StaticSparsityMatrixCSR{T, I, <:Vector, <:Vector, <:Vector}
    ) where {T, I}
    require_same_smoother_pattern(state, A)
    eltype(state) === T || throw(ArgumentError("Hybrid Gauss-Seidel coefficient type changed"))
    eltype(state.rowptr) === I || throw(ArgumentError("Hybrid Gauss-Seidel index type changed"))
    schedule = state.schedule
    foreach_hybrid_partition(schedule) do p
        @inbounds for t in schedule.offsets[p]:(schedule.offsets[p + 1] - 1)
            i = schedule.rows[t]
            position = schedule.diagonal_positions[i]
            diagonal = zero(T)
            if position > 0
                diagonal = A.nzval[position]
            else
                # Preserve duplicate-diagonal and unsorted-CSR semantics.
                for k in A.rowptr[i]:(A.rowptr[i + 1] - 1)
                    if A.colval[k] == i
                        diagonal += A.nzval[k]
                    end
                end
            end
            h = schedule.boundary_index[i]
            remote = zero(T)
            if state.config.l1 && h > 0
                @simd for entry in schedule.remote_offsets[h]:(schedule.remote_offsets[h + 1] - 1)
                    remote += abs(A.nzval[schedule.remote_positions[entry]])
                end
            end
            # Baker et al., scaled ℓ₁ GS (Remark 6.1, LLNL-TR-489114).
            # Retain the sign for consistently negative matrix rows. Defer
            # errors until after the batch so no worker throws inside @batch.
            scale = copysign(abs(diagonal) + remote / 2, diagonal)
            if isfinite(scale) && !iszero(diagonal)
                state.inverse_diagonal[i] = inv(scale)
            else
                state.inverse_diagonal[i] = T(NaN)
            end
        end
    end
    invalid = findfirst(v -> !isfinite(v) || iszero(v), state.inverse_diagonal)
    isnothing(invalid) || throw(ArgumentError("Hybrid Gauss-Seidel requires a finite nonzero diagonal and scaling in row $invalid"))
    state.matrix = A
    state.rowptr, state.colval = A.rowptr, A.colval
    return state
end

@inline function foreach_hybrid_partition(f, schedule)
    np = length(schedule.offsets) - 1
    if np > 1 && Threads.nthreads() > 1
        @batch minbatch = 1 for p in 1:np
            f(p)
        end
    else
        for p in 1:np
            f(p)
        end
    end
    return nothing
end

# Direction and correction mode are compile-time flags. The correction path reads
# only predecessors in the local triangular solve, so stale scratch and x are
# never read on the zero-initial path (including a buffer filled with NaNs).
@inline function hybrid_row_value(
        x, A, rhs, scratch, schedule, i,
        direction::Val, ::Val{correction}
    ) where {correction}
    if correction && schedule.triangular_ranges isa Val{true}
        return hybrid_triangular_value(x, A, rhs, schedule, i, direction)
    end
    h = @inbounds schedule.boundary_index[i]
    value = @inbounds rhs[i]
    if h == 0
        @inbounds @simd for k in A.rowptr[i]:(A.rowptr[i + 1] - 1)
            j = A.colval[k]
            if !correction || hybrid_predecessor(j, i, direction)
                value -= A.nzval[k] * x[j]
            end
        end
    else
        if !correction
            value -= @inbounds scratch[h]
        end
        @inbounds @simd for t in schedule.local_offsets[h]:(schedule.local_offsets[h + 1] - 1)
            k = schedule.local_positions[t]
            j = A.colval[k]
            if !correction || hybrid_predecessor(j, i, direction)
                value -= A.nzval[k] * x[j]
            end
        end
    end
    return value
end

@inline hybrid_predecessor(j, i, ::Val{true}) = j < i
@inline hybrid_predecessor(j, i, ::Val{false}) = j > i

@inline function hybrid_sweep_rows(lo, hi, ::Val{forward}) where {forward}
    if forward
        return lo:hi
    else
        return hi:-1:lo
    end
end

function hybrid_sweep!(x, A, b, state, direction::Val)
    schedule = state.schedule
    damping = smoother_damping(state)
    if length(schedule.offsets) == 2
        # Coarse AMG levels should pay no partition-map or interface overhead.
        rows = hybrid_sweep_rows(1, state.n, direction)
        return gauss_seidel_sweep!(x, A, b, state.inverse_diagonal, damping, rows)
    end
    # All reads of cross-partition x finish before any partition writes x.
    # Store the frozen remote matvec, rather than copying the entire iterate.
    if !isempty(schedule.boundary_rows)
        foreach_hybrid_partition(schedule) do p
            @inbounds for h in schedule.boundary_offsets[p]:(schedule.boundary_offsets[p + 1] - 1)
                value = zero(eltype(x))
                @simd for t in schedule.remote_offsets[h]:(schedule.remote_offsets[h + 1] - 1)
                    k = schedule.remote_positions[t]
                    value += A.nzval[k] * x[A.colval[k]]
                end
                state.halo_rhs[h] = value
            end
        end
    end
    foreach_hybrid_partition(schedule) do p
        lo, hi = schedule.offsets[p], schedule.offsets[p + 1] - 1
        order = hybrid_sweep_rows(lo, hi, direction)
        @inbounds for t in order
            i = schedule.rows[t]
            value = hybrid_row_value(x, A, b, state.halo_rhs, schedule, i, direction, Val(false))
            x[i] += damping * state.inverse_diagonal[i] * value
        end
    end
    return x
end

function hybrid_correction!(
        x, A, residual, state, direction::Val,
        ::Val{zero_initial}
    ) where {zero_initial}
    schedule, correction = state.schedule, state.correction
    damping = smoother_damping(state)
    foreach_hybrid_partition(schedule) do p
        lo, hi = schedule.offsets[p], schedule.offsets[p + 1] - 1
        order = hybrid_sweep_rows(lo, hi, direction)
        @inbounds for t in order
            i = schedule.rows[t]
            value = hybrid_row_value(correction, A, residual, state.halo_rhs, schedule, i, direction, Val(true))
            delta = damping * state.inverse_diagonal[i] * value
            correction[i] = delta
            if zero_initial
                x[i] = delta
            else
                x[i] += delta
            end
        end
    end
    return x
end

function hybrid_smooth!(
        x, A, b, state, steps, direction;
        residual = nothing, zero_initial::Bool = false
    )
    steps > 0 || throw(ArgumentError("smoothing steps must be positive"))
    check_smoother_dimensions(x, A, b)
    require_same_smoother_pattern(state, A)
    if !isnothing(residual) && length(residual) != state.n
        throw(DimensionMismatch("residual length does not match matrix rows"))
    end
    if x === b
        b = copy(b)
    end
    if zero_initial
        hybrid_correction!(x, A, b, state, direction, Val(true))
    elseif !isnothing(residual)
        hybrid_correction!(x, A, residual, state, direction, Val(false))
    else
        hybrid_sweep!(x, A, b, state, direction)
    end
    for _ in 2:steps
        hybrid_sweep!(x, A, b, state, direction)
    end
    return x
end

function smooth!(
        x::Vector, state::HybridGaussSeidelState,
        A::StaticSparsityMatrixCSR, b::Vector;
        steps::Integer = smoother_steps(state.config), zero_initial::Bool = false
    )
    return smooth_level!(x, A, b, state, Int(steps); zero_initial)
end

function smooth_level!(
        x::Vector, A::StaticSparsityMatrixCSR, b::Vector,
        state::HybridGaussSeidelState, steps::Int; kwargs...
    )
    return hybrid_smooth!(x, A, b, state, steps, Val(true); kwargs...)
end

function smooth_result!(
        x::Vector, A::StaticSparsityMatrixCSR, b::Vector,
        state::HybridGaussSeidelState, steps::Int; kwargs...
    )
    return hybrid_smooth!(x, A, b, state, steps, Val(false); kwargs...)
end

function apply!(x::Vector, state::HybridGaussSeidelState, b::Vector)
    require_apply_dimensions(x, state, b)
    if x === b
        b = copy(b)
    end
    A, steps = state.matrix, smoother_steps(state.config)
    smooth_level!(x, A, b, state, steps; zero_initial = true)
    return smooth_result!(x, A, b, state, steps)
end
