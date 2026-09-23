@inline function device_find_column(rowptr, colval, row, column)
    # CSR column ranges are sorted by construction. Find the first stored
    # column that is not smaller than the target.
    index_one = one(eltype(rowptr))
    @inbounds lo = rowptr[row]
    @inbounds hi = rowptr[row + 1] - index_one
    row_end = hi
    while lo < hi
        mid = (lo + hi) >>> 1
        @inbounds if colval[mid] < column
            lo = mid + index_one
        else
            hi = mid
        end
    end
    @inbounds found = lo <= row_end && colval[lo] == column
    return found ? lo : zero(eltype(rowptr))
end

function diagonal_positions(rowptr::Vector{Ti}, colval::Vector{Ti}, n::Int) where {Ti}
    positions = Vector{Ti}(undef, n)
    @inbounds for i in 1:n
        position = device_find_column(rowptr, colval, Ti(i), Ti(i))
        iszero(position) && throw(ArgumentError("ILU0 requires a stored diagonal in row $i"))
        positions[i] = position
    end
    return positions
end

function transpose_positions(rowptr::Vector{Ti}, colval::Vector{Ti}, n::Int) where {Ti}
    positions = zeros(Ti, length(colval))
    @inbounds for i in 1:n
        for k in rowptr[i]:(rowptr[i + 1] - one(Ti))
            j = colval[k]
            if j != i
                positions[k] = device_find_column(rowptr, colval, j, Ti(i))
            end
        end
    end
    return positions
end

function level_schedule(
        rowptr::Vector{Ti}, colval::Vector{Ti}, n::Int;
        upper::Bool = false
    ) where {Ti}
    levels = zeros(Int, n)
    indices = if upper
        n:-1:1
    else
        1:n
    end
    @inbounds for i in indices
        level = 1
        for k in rowptr[i]:(rowptr[i + 1] - one(Ti))
            j = Int(colval[k])
            dependency = if upper
                j > i
            else
                j < i
            end
            if dependency
                level = max(level, levels[j] + 1)
            end
        end
        levels[i] = level
    end
    number_of_levels = maximum(levels; init = 0)
    counts = zeros(Int, number_of_levels)
    @inbounds for level in levels
        counts[level] += 1
    end
    offsets = Vector{Int}(undef, number_of_levels + 1)
    offsets[1] = 1
    @inbounds for level in 1:number_of_levels
        offsets[level + 1] = offsets[level] + counts[level]
    end
    cursor = copy(offsets)
    rows = Vector{Ti}(undef, n)
    @inbounds for i in 1:n
        level = levels[i]
        rows[cursor[level]] = Ti(i)
        cursor[level] += 1
    end
    return offsets, rows
end

function ilu_symbolic(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    rowptr = host_prefix(A.rowptr, matrix_nrows(A) + 1)
    colval = host_prefix(A.colval, matrix_nonzeros(A))
    @inbounds for i in 1:matrix_nrows(A)
        issorted(view(colval, rowptr[i]:(rowptr[i + 1] - one(Ti)))) ||
            throw(ArgumentError("ILU0 requires sorted CSR columns"))
    end
    diagonal = diagonal_positions(rowptr, colval, matrix_nrows(A))
    transpose = transpose_positions(rowptr, colval, matrix_nrows(A))
    factor_offsets, factor_rows = level_schedule(rowptr, colval, matrix_nrows(A))
    upper_offsets, upper_rows = level_schedule(rowptr, colval, matrix_nrows(A); upper = true)
    return (;
        rowptr, colval, diagonal, transpose, factor_offsets, factor_rows,
        upper_offsets, upper_rows,
    )
end

@kernel function ilu0_factor_level_kernel!(
        factors, inverse_diagonal,
        @Const(rowptr), @Const(colval),
        @Const(diagonal_positions),
        @Const(rows), first, count
    )
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        row_end = rowptr[i + 1] - one(eltype(rowptr))
        @inbounds for k in rowptr[i]:row_end
            j = colval[k]
            if j < i
                multiplier = factors[k] * inverse_diagonal[j]
                factors[k] = multiplier
                for p in rowptr[j]:(rowptr[j + 1] - one(eltype(rowptr)))
                    column = colval[p]
                    if column > j
                        target = device_find_column(rowptr, colval, i, column)
                        if !iszero(target)
                            factors[target] -= multiplier * factors[p]
                        end
                    end
                end
            end
        end
        inverse_diagonal[i] = inv(factors[diagonal_positions[i]])
    end
end

@kernel function dilu_factor_level_kernel!(
        inverse_diagonal, @Const(values),
        @Const(rowptr), @Const(colval),
        @Const(diagonal_positions),
        @Const(transpose_positions),
        @Const(rows), first, count
    )
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        diagonal = values[diagonal_positions[i]]
        # D_i = A_ii - sum_{j < i} A_ij inv(D_j) A_ji. The comparison is
        # deliberately against the natural row index: the OPM row reordering
        # is a storage optimization and does not change the DILU ordering.
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            opposite = transpose_positions[k]
            if j < i && !iszero(opposite)
                diagonal -= values[k] * inverse_diagonal[j] * values[opposite]
            end
        end
        inverse_diagonal[i] = inv(diagonal)
    end
end

@kernel function ilu0_lower_level_kernel!(
        work, @Const(rhs), @Const(factors),
        @Const(rowptr), @Const(colval),
        @Const(rows), first, count
    )
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = rhs[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            j < i && (value -= factors[k] * work[j])
        end
        @inbounds work[i] = value
    end
end

@kernel function ilu0_upper_level_kernel!(
        x, @Const(work), @Const(factors),
        @Const(inverse_diagonal),
        @Const(rowptr), @Const(colval),
        damping, @Const(rows), first, count
    )
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = work[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            j > i && (value -= factors[k] * x[j])
        end
        @inbounds x[i] = damping * (inverse_diagonal[i] * value)
    end
end

@kernel function dilu_lower_level_kernel!(
        work, @Const(rhs), @Const(values),
        @Const(inverse_diagonal),
        @Const(rowptr), @Const(colval),
        @Const(rows), first, count
    )
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = rhs[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            j < i && (value -= values[k] * work[j])
        end
        @inbounds work[i] = inverse_diagonal[i] * value
    end
end

@kernel function dilu_upper_level_kernel!(
        x, @Const(work), @Const(values),
        @Const(inverse_diagonal),
        @Const(rowptr), @Const(colval),
        damping,
        @Const(rows), first, count
    )
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        correction = zero(eltype(x))
        diagonal_inverse = inverse_diagonal[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            j > i && (correction += values[k] * x[j])
        end
        @inbounds x[i] = damping * (work[i] - diagonal_inverse * correction)
    end
end

@kernel function ilu0_smooth_lower_level_kernel!(
        work, @Const(rhs),
        @Const(x), @Const(matrix_values), @Const(factors),
        @Const(rowptr), @Const(colval), @Const(rows), first, count
    )
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = rhs[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            value -= matrix_values[k] * x[j]
            j < i && (value -= factors[k] * work[j])
        end
        @inbounds work[i] = value
    end
end

@kernel function ilu0_smooth_upper_level_kernel!(
        x, correction, @Const(work),
        @Const(factors), @Const(inverse_diagonal), @Const(rowptr),
        @Const(colval), damping, @Const(rows), first, count
    )
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = work[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            j > i && (value -= factors[k] * correction[j])
        end
        delta = damping * (inverse_diagonal[i] * value)
        @inbounds begin
            correction[i] = delta
            x[i] += delta
        end
    end
end

@kernel function dilu_smooth_lower_level_kernel!(
        work, @Const(rhs),
        @Const(x), @Const(values), @Const(inverse_diagonal),
        @Const(rowptr), @Const(colval),
        @Const(rows), first, count
    )
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = rhs[i]
        diagonal_inverse = inverse_diagonal[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            value -= values[k] * x[j]
            j < i && (value -= values[k] * work[j])
        end
        @inbounds work[i] = diagonal_inverse * value
    end
end

@kernel function dilu_smooth_upper_level_kernel!(
        x, correction, @Const(work),
        @Const(values), @Const(inverse_diagonal), @Const(rowptr),
        @Const(colval), damping,
        @Const(rows), first, count
    )
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = work[i]
        diagonal_inverse = inverse_diagonal[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            j > i &&
                (value -= diagonal_inverse * values[k] * correction[j])
        end
        delta = damping * value
        @inbounds begin
            correction[i] = delta
            x[i] += delta
        end
    end
end

function allocate_factor_storage(A::StaticSparsityMatrixCSR{Tv}) where {Tv}
    factors = KernelAbstractions.allocate(matrix_backend(A), Tv, matrix_nonzeros(A))
    inverse_diagonal = KernelAbstractions.allocate(matrix_backend(A), Tv, matrix_nrows(A))
    return factors, inverse_diagonal
end

smoother_vector_eltype(::Type{T}) where {T <: Number} = T
smoother_vector_eltype(::Type{<:StaticMatrix{N, N, T}}) where {N, T} =
    SVector{N, T}

function allocate_ilu_work(A::StaticSparsityMatrixCSR{Tv}) where {Tv}
    backend = matrix_backend(A)
    vector_eltype = smoother_vector_eltype(Tv)
    work = KernelAbstractions.allocate(
        backend, vector_eltype, matrix_nrows(A)
    )
    residual = similar(work)
    return work, residual
end

function setup_ilu0_kernels(
        A, factors, inverse_diagonal, diagonal_positions,
        factor_offsets, factor_rows, upper_offsets, upper_rows,
        work, residual, damping
    )
    length(factor_offsets) > 1 || return (
        nothing, (nothing, nothing), (nothing, nothing)
    )
    backend = matrix_backend(A)
    block_size = matrix_batch_size(A)
    lower_first = factor_offsets[1]
    lower_count = factor_offsets[2] - lower_first
    upper_first = upper_offsets[1]
    upper_count = upper_offsets[2] - upper_first
    factor = setup_smoother_kernel(
        ilu0_factor_level_kernel!, backend, block_size,
        factors, inverse_diagonal, A.rowptr, A.colval, diagonal_positions,
        factor_rows, lower_first, lower_count;
        ndrange = lower_count,
        number_of_launches = length(factor_offsets) - 1
    )
    lower = setup_smoother_kernel(
        ilu0_lower_level_kernel!, backend, block_size,
        work, work, factors, A.rowptr, A.colval,
        factor_rows, lower_first, lower_count;
        ndrange = lower_count,
        number_of_launches = length(factor_offsets) - 1
    )
    upper = setup_smoother_kernel(
        ilu0_upper_level_kernel!, backend, block_size,
        residual, work, factors, inverse_diagonal, A.rowptr, A.colval,
        damping,
        upper_rows, upper_first, upper_count;
        ndrange = upper_count,
        number_of_launches = length(upper_offsets) - 1
    )
    smooth_lower = setup_smoother_kernel(
        ilu0_smooth_lower_level_kernel!, backend, block_size,
        work, work, residual, A.nzval, factors, A.rowptr, A.colval,
        factor_rows, lower_first, lower_count;
        ndrange = lower_count,
        number_of_launches = length(factor_offsets) - 1
    )
    smooth_upper = setup_smoother_kernel(
        ilu0_smooth_upper_level_kernel!, backend, block_size,
        residual, residual, work, factors, inverse_diagonal,
        A.rowptr, A.colval, damping,
        upper_rows, upper_first, upper_count;
        ndrange = upper_count,
        number_of_launches = length(upper_offsets) - 1
    )
    return factor, (lower, upper), (smooth_lower, smooth_upper)
end

function setup_dilu_kernels(
        A, values, inverse_diagonal, diagonal_positions,
        transpose_positions, factor_offsets, factor_rows,
        upper_offsets, upper_rows, work, residual, damping
    )
    length(factor_offsets) > 1 || return (
        nothing, (nothing, nothing), (nothing, nothing)
    )
    backend = matrix_backend(A)
    block_size = matrix_batch_size(A)
    lower_first = factor_offsets[1]
    lower_count = factor_offsets[2] - lower_first
    upper_first = upper_offsets[1]
    upper_count = upper_offsets[2] - upper_first
    factor = setup_smoother_kernel(
        dilu_factor_level_kernel!, backend, block_size,
        inverse_diagonal, values, A.rowptr, A.colval,
        diagonal_positions, transpose_positions,
        factor_rows, lower_first, lower_count;
        ndrange = lower_count,
        number_of_launches = length(factor_offsets) - 1
    )
    lower = setup_smoother_kernel(
        dilu_lower_level_kernel!, backend, block_size,
        work, work, values, inverse_diagonal, A.rowptr, A.colval,
        factor_rows, lower_first, lower_count;
        ndrange = lower_count,
        number_of_launches = length(factor_offsets) - 1
    )
    upper = setup_smoother_kernel(
        dilu_upper_level_kernel!, backend, block_size,
        residual, work, values, inverse_diagonal, A.rowptr, A.colval,
        damping,
        upper_rows, upper_first, upper_count;
        ndrange = upper_count,
        number_of_launches = length(upper_offsets) - 1
    )
    smooth_lower = setup_smoother_kernel(
        dilu_smooth_lower_level_kernel!, backend, block_size,
        work, work, residual, values, inverse_diagonal, A.rowptr, A.colval,
        factor_rows, lower_first, lower_count;
        ndrange = lower_count,
        number_of_launches = length(factor_offsets) - 1
    )
    smooth_upper = setup_smoother_kernel(
        dilu_smooth_upper_level_kernel!, backend, block_size,
        residual, residual, work, values, inverse_diagonal,
        A.rowptr, A.colval, damping,
        upper_rows, upper_first, upper_count;
        ndrange = upper_count,
        number_of_launches = length(upper_offsets) - 1
    )
    return factor, (lower, upper), (smooth_lower, smooth_upper)
end

replaced_ilu_storage_fields(::ILU0State) = (
    :factors, :inverse_diagonal, :diagonal_positions, :factor_rows,
    :upper_rows, :work, :residual,
)
replaced_ilu_storage_fields(::DILUState) = (
    :values, :inverse_diagonal, :diagonal_positions, :transpose_positions,
    :factor_rows, :upper_rows, :work, :residual,
)

function replaced_ilu_storage_bytes(state::Union{ILU0State, DILUState})
    bytes = 0
    for field in replaced_ilu_storage_fields(state)
        bytes += backend_buffer_bytes(getproperty(state, field))
    end
    return bytes
end

replaced_ilu_storage_bytes(state) = 0

function reuse_ilu_smoother!(reuse, A, config, ::Type{S}) where {S}
    compatible = reuse isa S &&
        same_backend(reuse.backend, matrix_backend(A)) &&
        same_smoother_pattern(reuse, A)
    compatible || return nothing
    reuse.config = config
    return update_smoother!(reuse, A)
end

function setup_smoother(
        A::StaticSparsityMatrixCSR, config::ILU0;
        reuse = nothing, reallocation_tracker = nothing
    )
    require_square_matrix(A, "ILU0")
    reused = reuse_ilu_smoother!(reuse, A, config, ILU0State)
    isnothing(reused) || return reused
    mark_backend_reallocation!(
        reallocation_tracker, replaced_ilu_storage_bytes(reuse)
    )
    symbolic = ilu_symbolic(A)
    backend = matrix_backend(A)
    factors, inverse_diagonal = allocate_factor_storage(A)
    work, residual = allocate_ilu_work(A)
    diagonal_positions = backend_copy(backend, symbolic.diagonal)
    factor_rows = backend_copy(backend, symbolic.factor_rows)
    upper_rows = backend_copy(backend, symbolic.upper_rows)
    damping = convert(
        matrix_real_type(eltype(inverse_diagonal)), config.damping
    )
    factor_kernel, solve_kernels, smooth_kernels = setup_ilu0_kernels(
        A, factors, inverse_diagonal, diagonal_positions,
        symbolic.factor_offsets, factor_rows,
        symbolic.upper_offsets, upper_rows, work, residual, damping
    )
    state = ILU0State(
        factors, inverse_diagonal, work, residual,
        A.rowptr, A.colval,
        diagonal_positions,
        symbolic.factor_offsets, factor_rows,
        symbolic.upper_offsets, upper_rows,
        symbolic.rowptr, symbolic.colval,
        factor_kernel, solve_kernels, smooth_kernels, config,
        backend, matrix_batch_size(A), matrix_nrows(A)
    )
    return update_smoother!(state, A)
end

function setup_smoother(
        A::StaticSparsityMatrixCSR, config::DILU;
        reuse = nothing, reallocation_tracker = nothing
    )
    require_square_matrix(A, "DILU")
    reused = reuse_ilu_smoother!(reuse, A, config, DILUState)
    isnothing(reused) || return reused
    mark_backend_reallocation!(
        reallocation_tracker, replaced_ilu_storage_bytes(reuse)
    )
    symbolic = ilu_symbolic(A)
    backend = matrix_backend(A)
    # The dependency levels expose parallelism without changing the natural
    # ordering used by the DILU recurrence and triangular solves.
    values, inverse_diagonal = allocate_factor_storage(A)
    work, residual = allocate_ilu_work(A)
    diagonal_positions = backend_copy(backend, symbolic.diagonal)
    transpose_positions = backend_copy(backend, symbolic.transpose)
    factor_rows = backend_copy(backend, symbolic.factor_rows)
    upper_rows = backend_copy(backend, symbolic.upper_rows)
    damping = convert(
        matrix_real_type(eltype(inverse_diagonal)), config.damping
    )
    factor_kernel, solve_kernels, smooth_kernels = setup_dilu_kernels(
        A, values, inverse_diagonal, diagonal_positions, transpose_positions,
        symbolic.factor_offsets, factor_rows,
        symbolic.upper_offsets, upper_rows, work, residual, damping
    )
    state = DILUState(
        inverse_diagonal, work, residual, values,
        A.rowptr, A.colval,
        diagonal_positions, transpose_positions,
        symbolic.factor_offsets, factor_rows,
        symbolic.upper_offsets, upper_rows,
        symbolic.rowptr, symbolic.colval,
        factor_kernel, solve_kernels, smooth_kernels, config,
        backend, matrix_batch_size(A), matrix_nrows(A)
    )
    return update_smoother!(state, A)
end

function update_smoother!(state::ILU0State, A::StaticSparsityMatrixCSR)
    require_same_smoother_pattern(state, A)
    copyto!(state.factors, 1, A.nzval, 1, matrix_nonzeros(A))
    kernel = state.factor_kernel
    launch_factor_levels!(state, kernel)
    return state
end

function launch_factor_levels!(state::ILU0State, kernel)
    for level in 1:(length(state.factor_offsets) - 1)
        first = state.factor_offsets[level]
        count = state.factor_offsets[level + 1] - first
        launch_smoother_kernel(
            kernel,
            state.factors, state.inverse_diagonal,
            state.rowptr, state.colval, state.diagonal_positions,
            state.factor_rows, first, count;
            ndrange = count, launch_index = level
        )
    end
    return state
end

function update_smoother!(state::DILUState, A::StaticSparsityMatrixCSR)
    require_same_smoother_pattern(state, A)
    copyto!(state.values, 1, A.nzval, 1, matrix_nonzeros(A))
    kernel = state.factor_kernel
    launch_factor_levels!(state, kernel)
    return state
end

function launch_factor_levels!(state::DILUState, kernel)
    for level in 1:(length(state.factor_offsets) - 1)
        first = state.factor_offsets[level]
        count = state.factor_offsets[level + 1] - first
        launch_smoother_kernel(
            kernel,
            state.inverse_diagonal, state.values,
            state.rowptr, state.colval, state.diagonal_positions,
            state.transpose_positions, state.factor_rows, first, count;
            ndrange = count, launch_index = level
        )
    end
    return state
end

function update_smoother!(
        state::ILU0State{<:Vector, <:Vector, <:Vector, <:Vector},
        A::StaticSparsityMatrixCSR{Tv, Ti, <:Vector, <:Vector, <:Vector}
    ) where {Tv, Ti}
    require_same_smoother_pattern(state, A)
    copyto!(state.factors, 1, A.nzval, 1, matrix_nonzeros(A))
    factors = state.factors
    inverse_diagonal = state.inverse_diagonal
    rowptr = state.rowptr
    colval = state.colval
    diagonal = state.diagonal_positions
    rows = state.factor_rows
    offsets = state.factor_offsets
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        @batch for q in 1:count
            i = rows[first + q - 1]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(Ti))
                j = colval[k]
                if j < i
                    multiplier = factors[k] * inverse_diagonal[j]
                    factors[k] = multiplier
                    for p in rowptr[j]:(rowptr[j + 1] - one(Ti))
                        column = colval[p]
                        if column > j
                            target = device_find_column(
                                rowptr, colval, i, column
                            )
                            !iszero(target) &&
                                (factors[target] -= multiplier * factors[p])
                        end
                    end
                end
            end
            inverse_diagonal[i] = inv(factors[diagonal[i]])
        end
    end
    return state
end

function update_smoother!(
        state::DILUState{<:Vector, <:Vector, <:Vector, <:Vector},
        A::StaticSparsityMatrixCSR{Tv, Ti, <:Vector, <:Vector, <:Vector}
    ) where {Tv, Ti}
    require_same_smoother_pattern(state, A)
    copyto!(state.values, 1, A.nzval, 1, matrix_nonzeros(A))
    values = state.values
    inverse_diagonal = state.inverse_diagonal
    rowptr = state.rowptr
    colval = state.colval
    diagonal_positions = state.diagonal_positions
    transpose_positions = state.transpose_positions
    rows = state.factor_rows
    offsets = state.factor_offsets
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        @batch for q in 1:count
            i = rows[first + q - 1]
            diagonal = values[diagonal_positions[i]]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(Ti))
                j = colval[k]
                opposite = transpose_positions[k]
                if j < i && !iszero(opposite)
                    diagonal -= values[k] * inverse_diagonal[j] * values[opposite]
                end
            end
            inverse_diagonal[i] = inv(diagonal)
        end
    end
    return state
end

function ilu_solve!(x, state::ILU0State, b)
    ensure_smoother_work!(state, b)
    kernels = state.solve_kernels
    return ilu_solve_levels!(x, state, b, state.work, kernels)
end

function ilu_solve_levels!(x, state::ILU0State, b, work, kernels)
    lower, upper = kernels
    for level in 1:(length(state.factor_offsets) - 1)
        first = state.factor_offsets[level]
        count = state.factor_offsets[level + 1] - first
        launch_smoother_kernel(
            lower,
            work, b, state.factors, state.rowptr, state.colval,
            state.factor_rows, first, count;
            ndrange = count, launch_index = level
        )
    end
    for level in 1:(length(state.upper_offsets) - 1)
        first = state.upper_offsets[level]
        count = state.upper_offsets[level + 1] - first
        launch_smoother_kernel(
            upper,
            x, work, state.factors, state.inverse_diagonal,
            state.rowptr, state.colval, smoother_damping(state),
            state.upper_rows, first, count;
            ndrange = count, launch_index = level
        )
    end
    return x
end

function ilu_solve!(x, state::DILUState, b)
    ensure_smoother_work!(state, b)
    kernels = state.solve_kernels
    return ilu_solve_levels!(x, state, b, state.work, kernels)
end

function ilu_solve_levels!(x, state::DILUState, b, work, kernels)
    lower, upper = kernels
    for level in 1:(length(state.factor_offsets) - 1)
        first = state.factor_offsets[level]
        count = state.factor_offsets[level + 1] - first
        launch_smoother_kernel(
            lower,
            work, b, state.values, state.inverse_diagonal,
            state.rowptr, state.colval,
            state.factor_rows, first, count;
            ndrange = count, launch_index = level
        )
    end
    for level in 1:(length(state.upper_offsets) - 1)
        first = state.upper_offsets[level]
        count = state.upper_offsets[level + 1] - first
        launch_smoother_kernel(
            upper,
            x, work, state.values, state.inverse_diagonal,
            state.rowptr, state.colval, smoother_damping(state),
            state.upper_rows, first, count;
            ndrange = count, launch_index = level
        )
    end
    return x
end

function ilu_solve!(
        x::AbstractVector,
        state::ILU0State{F, D, RP, CV},
        b::AbstractVector
    ) where {F, D, RP <: Vector, CV}
    ensure_smoother_work!(state, b)
    return ilu_solve_cpu!(x, state, b, state.work)
end

function ilu_solve_cpu!(x, state::ILU0State, b, work)
    factors = state.factors
    inverse_diagonal = state.inverse_diagonal
    rowptr = state.rowptr
    colval = state.colval
    rows = state.factor_rows
    offsets = state.factor_offsets
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        @batch for q in 1:count
            i = rows[first + q - 1]
            value = b[i]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
                j = colval[k]
                j < i && (value -= factors[k] * work[j])
            end
            @inbounds work[i] = value
        end
    end
    damping = smoother_damping(state)
    rows = state.upper_rows
    offsets = state.upper_offsets
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        @batch for q in 1:count
            i = rows[first + q - 1]
            value = work[i]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
                j = colval[k]
                j > i && (value -= factors[k] * x[j])
            end
            @inbounds x[i] = damping * (inverse_diagonal[i] * value)
        end
    end
    return x
end

function ilu_solve!(
        x::AbstractVector,
        state::DILUState{D, AV, RP, CV},
        b::AbstractVector
    ) where {D, AV, RP <: Vector, CV}
    ensure_smoother_work!(state, b)
    return ilu_solve_cpu!(x, state, b, state.work)
end

function ilu_solve_cpu!(x, state::DILUState, b, work)
    values = state.values
    inverse_diagonal = state.inverse_diagonal
    rowptr = state.rowptr
    colval = state.colval
    rows = state.factor_rows
    offsets = state.factor_offsets
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        @batch for q in 1:count
            i = rows[first + q - 1]
            value = b[i]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
                j = colval[k]
                j < i && (value -= values[k] * work[j])
            end
            @inbounds work[i] = inverse_diagonal[i] * value
        end
    end
    damping = smoother_damping(state)
    rows = state.upper_rows
    offsets = state.upper_offsets
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        @batch for q in 1:count
            i = rows[first + q - 1]
            correction = zero(eltype(x))
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
                j = colval[k]
                j > i && (correction += values[k] * x[j])
            end
            @inbounds x[i] = damping * (
                work[i] - inverse_diagonal[i] * correction
            )
        end
    end
    return x
end

function ilu_smooth_result!(x, A::StaticSparsityMatrixCSR, b, state::ILU0State)
    ensure_smoother_work!(state, b)
    kernels = state.smooth_kernels
    return ilu_smooth_levels!(
        x, A, b, state, state.work, state.residual, kernels
    )
end

function ilu_smooth_levels!(x, A, b, state::ILU0State, work, correction, kernels)
    lower, upper = kernels
    for level in 1:(length(state.factor_offsets) - 1)
        first = state.factor_offsets[level]
        count = state.factor_offsets[level + 1] - first
        launch_smoother_kernel(
            lower,
            work, b, x, A.nzval, state.factors,
            state.rowptr, state.colval,
            state.factor_rows, first, count;
            ndrange = count, launch_index = level
        )
    end
    for level in 1:(length(state.upper_offsets) - 1)
        first = state.upper_offsets[level]
        count = state.upper_offsets[level + 1] - first
        launch_smoother_kernel(
            upper,
            x, correction, work,
            state.factors, state.inverse_diagonal,
            state.rowptr, state.colval, smoother_damping(state),
            state.upper_rows, first, count;
            ndrange = count, launch_index = level
        )
    end
    return x
end

function ilu_smooth_result!(x, A::StaticSparsityMatrixCSR, b, state::DILUState)
    ensure_smoother_work!(state, b)
    kernels = state.smooth_kernels
    return ilu_smooth_levels!(
        x, A, b, state, state.work, state.residual, kernels
    )
end

function ilu_smooth_levels!(x, A, b, state::DILUState, work, correction, kernels)
    lower, upper = kernels
    for level in 1:(length(state.factor_offsets) - 1)
        first = state.factor_offsets[level]
        count = state.factor_offsets[level + 1] - first
        launch_smoother_kernel(
            lower,
            work, b, x, state.values, state.inverse_diagonal,
            state.rowptr, state.colval,
            state.factor_rows, first, count;
            ndrange = count, launch_index = level
        )
    end
    for level in 1:(length(state.upper_offsets) - 1)
        first = state.upper_offsets[level]
        count = state.upper_offsets[level + 1] - first
        launch_smoother_kernel(
            upper,
            x, correction, work,
            state.values, state.inverse_diagonal,
            state.rowptr, state.colval, smoother_damping(state),
            state.upper_rows, first, count;
            ndrange = count, launch_index = level
        )
    end
    return x
end

function ilu_smooth_result!(
        x::AbstractVector, A::StaticSparsityMatrixCSR,
        b::AbstractVector,
        state::ILU0State{F, D, RP, CV}
    ) where {F, D, RP <: Vector, CV}
    ensure_smoother_work!(state, b)
    return ilu_smooth_result_cpu!(x, A, b, state, state.work, state.residual)
end

function ilu_smooth_result_cpu!(x, A, b, state::ILU0State, work, correction)
    factors = state.factors
    inverse_diagonal = state.inverse_diagonal
    rowptr = state.rowptr
    colval = state.colval
    rows = state.factor_rows
    offsets = state.factor_offsets
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        @batch for q in 1:count
            i = rows[first + q - 1]
            value = b[i]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
                j = colval[k]
                value -= A.nzval[k] * x[j]
                j < i && (value -= factors[k] * work[j])
            end
            @inbounds work[i] = value
        end
    end
    damping = smoother_damping(state)
    rows = state.upper_rows
    offsets = state.upper_offsets
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        @batch for q in 1:count
            i = rows[first + q - 1]
            value = work[i]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
                j = colval[k]
                j > i && (value -= factors[k] * correction[j])
            end
            delta = damping * (inverse_diagonal[i] * value)
            @inbounds begin
                correction[i] = delta
                x[i] += delta
            end
        end
    end
    return x
end

function ilu_smooth_result!(
        x::AbstractVector, A::StaticSparsityMatrixCSR,
        b::AbstractVector,
        state::DILUState{D, AV, RP, CV}
    ) where {D, AV, RP <: Vector, CV}
    ensure_smoother_work!(state, b)
    return ilu_smooth_result_cpu!(x, A, b, state, state.work, state.residual)
end

function ilu_smooth_result_cpu!(x, A, b, state::DILUState, work, correction)
    values = state.values
    inverse_diagonal = state.inverse_diagonal
    rowptr = state.rowptr
    colval = state.colval
    rows = state.factor_rows
    offsets = state.factor_offsets
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        @batch for q in 1:count
            i = rows[first + q - 1]
            value = b[i]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
                j = colval[k]
                value -= A.nzval[k] * x[j]
                j < i && (value -= values[k] * work[j])
            end
            @inbounds work[i] = inverse_diagonal[i] * value
        end
    end
    damping = smoother_damping(state)
    rows = state.upper_rows
    offsets = state.upper_offsets
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        @batch for q in 1:count
            i = rows[first + q - 1]
            value = work[i]
            diagonal_inverse = inverse_diagonal[i]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
                j = colval[k]
                j > i && (
                    value -= diagonal_inverse * values[k] * correction[j]
                )
            end
            delta = damping * value
            @inbounds begin
                correction[i] = delta
                x[i] += delta
            end
        end
    end
    return x
end

function apply!(
        x::AbstractVector, state::Union{ILU0State, DILUState},
        b::AbstractVector
    )
    require_apply_dimensions(x, state, b)
    same_backend(KernelAbstractions.get_backend(x), state.backend) ||
        throw(ArgumentError("output and smoother must use the same backend"))
    return ilu_solve!(x, state, b)
end

function apply_correction!(x, state::Union{ILU0State, DILUState}, residual)
    ilu_solve!(state.residual, state, residual)
    axpy!(
        x, state.residual, one(smoother_damping(state)), state.backend,
        state.block_size
    )
    return x
end

function smooth_level!(
        x, A::StaticSparsityMatrixCSR, b, state::Union{ILU0State, DILUState},
        steps::Int; residual = nothing, zero_initial::Bool = false
    )
    start = 1
    if !isnothing(residual)
        if zero_initial
            apply!(x, state, residual)
        else
            apply_correction!(x, state, residual)
        end
        start = 2
    elseif zero_initial
        fill_backend!(x, zero(eltype(x)), state.backend, state.block_size)
    end
    if start <= steps
        smooth!(x, state, A, b; steps = steps - start + 1)
    end
    return x
end

function smooth_result!(
        x, A::StaticSparsityMatrixCSR, b,
        state::Union{ILU0State, DILUState}, steps::Int;
        residual = nothing, zero_initial::Bool = false
    )
    steps > 0 || throw(ArgumentError("smoothing steps must be positive"))
    if isnothing(residual) && !zero_initial
        ilu_smooth_result!(x, A, b, state)
        if steps > 1
            smooth!(x, state, A, b; steps = steps - 1)
        end
        return x
    end
    return smooth_level!(
        x, A, b, state, steps;
        residual = residual, zero_initial = zero_initial
    )
end
