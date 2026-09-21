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

function diagonal_positions(rowptr::Vector{Ti}, colval::Vector{Ti}, n::Int) where Ti
    positions = Vector{Ti}(undef, n)
    @inbounds for i in 1:n
        position = device_find_column(rowptr, colval, Ti(i), Ti(i))
        iszero(position) && throw(ArgumentError("ILU0 requires a stored diagonal in row $i"))
        positions[i] = position
    end
    positions
end

function transpose_positions(rowptr::Vector{Ti}, colval::Vector{Ti}, n::Int) where Ti
    positions = zeros(Ti, length(colval))
    @inbounds for i in 1:n
        for k in rowptr[i]:(rowptr[i + 1] - one(Ti))
            j = colval[k]
            if j != i
                positions[k] = device_find_column(rowptr, colval, j, Ti(i))
            end
        end
    end
    positions
end

function level_schedule(rowptr::Vector{Ti}, colval::Vector{Ti}, n::Int;
                         upper::Bool=false) where Ti
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
    number_of_levels = maximum(levels; init=0)
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
    offsets, rows
end

function ilu_symbolic(A::StaticSparsityMatrixCSR{Tv,Ti}) where {Tv,Ti}
    rowptr = host_prefix(A.rowptr, matrix_nrows(A) + 1)
    colval = host_prefix(A.colval, matrix_nonzeros(A))
    @inbounds for i in 1:matrix_nrows(A)
        issorted(view(colval, rowptr[i]:(rowptr[i + 1] - one(Ti)))) ||
            throw(ArgumentError("ILU0 requires sorted CSR columns"))
    end
    diagonal = diagonal_positions(rowptr, colval, matrix_nrows(A))
    transpose = transpose_positions(rowptr, colval, matrix_nrows(A))
    factor_offsets, factor_rows = level_schedule(rowptr, colval, matrix_nrows(A))
    upper_offsets, upper_rows = level_schedule(rowptr, colval, matrix_nrows(A); upper=true)
    (; rowptr, colval, diagonal, transpose, factor_offsets, factor_rows,
       upper_offsets, upper_rows)
end

@kernel function ilu0_factor_level_kernel!(factors, inverse_diagonal,
                                            @Const(rowptr), @Const(colval),
                                            @Const(diagonal_positions),
                                            @Const(rows), first, count)
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

@kernel function dilu_factor_level_kernel!(inverse_diagonal, @Const(values),
                                            @Const(rowptr), @Const(colval),
                                            @Const(diagonal_positions),
                                            @Const(transpose_positions),
                                            @Const(rows), first, count)
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

@kernel function ilu0_lower_level_kernel!(work, @Const(rhs), @Const(factors),
                                           @Const(rowptr), @Const(colval),
                                           @Const(rows), first, count)
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

@kernel function ilu0_upper_level_kernel!(x, @Const(work), @Const(factors),
                                           @Const(inverse_diagonal),
                                           @Const(rowptr), @Const(colval),
                                           damping, @Const(rows), first, count)
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

@kernel function dilu_lower_level_kernel!(work, @Const(rhs), @Const(values),
                                           @Const(inverse_diagonal),
                                           @Const(rowptr), @Const(colval),
                                           @Const(rows), first, count)
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

@kernel function dilu_upper_level_kernel!(x, @Const(work), @Const(values),
                                           @Const(inverse_diagonal),
                                           @Const(rowptr), @Const(colval),
                                           damping,
                                           @Const(rows), first, count)
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

@kernel function ilu0_smooth_lower_level_kernel!(work, @Const(rhs),
        @Const(x), @Const(matrix_values), @Const(factors),
        @Const(rowptr), @Const(colval), @Const(rows), first, count)
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

@kernel function ilu0_smooth_upper_level_kernel!(x, correction, @Const(work),
        @Const(factors), @Const(inverse_diagonal), @Const(rowptr),
        @Const(colval), damping, @Const(rows), first, count)
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

@kernel function dilu_smooth_lower_level_kernel!(work, @Const(rhs),
        @Const(x), @Const(values), @Const(inverse_diagonal),
        @Const(rowptr), @Const(colval),
        @Const(rows), first, count)
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

@kernel function dilu_smooth_upper_level_kernel!(x, correction, @Const(work),
        @Const(values), @Const(inverse_diagonal), @Const(rowptr),
        @Const(colval), damping,
        @Const(rows), first, count)
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

function launch_levels!(kernel!, offsets, rows, arguments...)
    for level in 1:(length(offsets) - 1)
        first = offsets[level]
        count = offsets[level + 1] - first
        kernel!(arguments..., rows, first, count; ndrange=count)
    end
    nothing
end

function allocate_factor_storage(A::StaticSparsityMatrixCSR{Tv}) where Tv
    factors = KernelAbstractions.allocate(matrix_backend(A), Tv, matrix_nonzeros(A))
    inverse_diagonal = KernelAbstractions.allocate(matrix_backend(A), Tv, matrix_nrows(A))
    factors, inverse_diagonal
end

replaced_ilu_storage_fields(::ILU0State) = (
    :factors, :inverse_diagonal, :diagonal_positions, :factor_rows,
    :upper_rows, :work, :residual)
replaced_ilu_storage_fields(::DILUState) = (
    :values, :inverse_diagonal, :diagonal_positions, :transpose_positions,
    :factor_rows, :upper_rows, :work, :residual)

function replaced_ilu_storage_bytes(state::Union{ILU0State,DILUState})
    bytes = 0
    for field in replaced_ilu_storage_fields(state)
        bytes += backend_buffer_bytes(getproperty(state, field))
    end
    return bytes
end

replaced_ilu_storage_bytes(state) = 0

function reuse_ilu_smoother!(reuse, A, config, ::Type{S}) where S
    compatible = reuse isa S &&
                 same_backend(reuse.backend, matrix_backend(A)) &&
                 same_smoother_pattern(reuse, A)
    compatible || return nothing
    reuse.config = config
    return update_smoother!(reuse, A)
end

function setup_smoother(A::StaticSparsityMatrixCSR, config::ILU0;
        reuse=nothing, reallocation_tracker=nothing)
    require_square_matrix(A, "ILU0")
    reused = reuse_ilu_smoother!(reuse, A, config, ILU0State)
    isnothing(reused) || return reused
    mark_backend_reallocation!(
        reallocation_tracker, replaced_ilu_storage_bytes(reuse))
    symbolic = ilu_symbolic(A)
    factors, inverse_diagonal = allocate_factor_storage(A)
    state = ILU0State(factors, inverse_diagonal, nothing, nothing,
                      A.rowptr, A.colval,
                      backend_copy(matrix_backend(A), symbolic.diagonal),
                      symbolic.factor_offsets,
                      backend_copy(matrix_backend(A), symbolic.factor_rows),
                      symbolic.upper_offsets,
                      backend_copy(matrix_backend(A), symbolic.upper_rows),
                      symbolic.rowptr, symbolic.colval, config,
                      matrix_backend(A), matrix_batch_size(A), matrix_nrows(A))
    update_smoother!(state, A)
end

function setup_smoother(A::StaticSparsityMatrixCSR, config::DILU;
        reuse=nothing, reallocation_tracker=nothing)
    require_square_matrix(A, "DILU")
    reused = reuse_ilu_smoother!(reuse, A, config, DILUState)
    isnothing(reused) || return reused
    mark_backend_reallocation!(
        reallocation_tracker, replaced_ilu_storage_bytes(reuse))
    symbolic = ilu_symbolic(A)
    backend = matrix_backend(A)
    # The dependency levels expose parallelism without changing the natural
    # ordering used by the DILU recurrence and triangular solves.
    values, inverse_diagonal = allocate_factor_storage(A)
    state = DILUState(inverse_diagonal, nothing, nothing, values,
                      A.rowptr, A.colval,
                      backend_copy(backend, symbolic.diagonal),
                      backend_copy(backend, symbolic.transpose),
                      symbolic.factor_offsets,
                      backend_copy(backend, symbolic.factor_rows),
                      symbolic.upper_offsets,
                      backend_copy(backend, symbolic.upper_rows),
                      symbolic.rowptr, symbolic.colval, config,
                      backend, matrix_batch_size(A), matrix_nrows(A))
    update_smoother!(state, A)
end

factor_values(state::ILU0State) = state.factors
factor_values(state::DILUState) = state.values
factor_kernel(::ILU0State) = ilu0_factor_level_kernel!
factor_kernel(::DILUState) = dilu_factor_level_kernel!
factor_arguments(state::ILU0State) = (
    state.factors, state.inverse_diagonal, state.rowptr, state.colval,
    state.diagonal_positions)
factor_arguments(state::DILUState) = (
    state.inverse_diagonal, state.values, state.rowptr, state.colval,
    state.diagonal_positions, state.transpose_positions)

function update_smoother!(state::Union{ILU0State,DILUState},
                          A::StaticSparsityMatrixCSR)
    require_same_smoother_pattern(state, A)
    copyto!(factor_values(state), 1, A.nzval, 1, matrix_nonzeros(A))
    kernel! = factor_kernel(state)(state.backend, state.block_size)
    launch_levels!(kernel!, state.factor_offsets, state.factor_rows,
                   factor_arguments(state)...)
    state
end

function update_smoother!(
        state::ILU0State{<:Vector,<:Vector,<:Vector,<:Vector},
        A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector}
    ) where {Tv,Ti}
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
        foreach_cpu_row(count, state.block_size) do q
            i = rows[first + q - 1]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(Ti))
                j = colval[k]
                if j < i
                    multiplier = factors[k]*inverse_diagonal[j]
                    factors[k] = multiplier
                    for p in rowptr[j]:(rowptr[j + 1] - one(Ti))
                        column = colval[p]
                        if column > j
                            target = device_find_column(
                                rowptr, colval, i, column)
                            !iszero(target) &&
                                (factors[target] -= multiplier*factors[p])
                        end
                    end
                end
            end
            inverse_diagonal[i] = inv(factors[diagonal[i]])
        end
    end
    state
end

function update_smoother!(
        state::DILUState{<:Vector,<:Vector,<:Vector,<:Vector},
        A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector}
    ) where {Tv,Ti}
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
        foreach_cpu_row(count, state.block_size) do q
            i = rows[first + q - 1]
            diagonal = values[diagonal_positions[i]]
            @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(Ti))
                j = colval[k]
                opposite = transpose_positions[k]
                if j < i && !iszero(opposite)
                    diagonal -= values[k]*inverse_diagonal[j]*values[opposite]
                end
            end
            inverse_diagonal[i] = inv(diagonal)
        end
    end
    state
end

solve_kernels(::ILU0State) =
    (ilu0_lower_level_kernel!, ilu0_upper_level_kernel!)
solve_kernels(::DILUState) =
    (dilu_lower_level_kernel!, dilu_upper_level_kernel!)
lower_solve_arguments(state::ILU0State, b) = (
    state.work, b, state.factors, state.rowptr, state.colval)
lower_solve_arguments(state::DILUState, b) = (
    state.work, b, state.values, state.inverse_diagonal,
    state.rowptr, state.colval)
upper_solve_arguments(state::ILU0State, x) = (
    x, state.work, state.factors, state.inverse_diagonal,
    state.rowptr, state.colval, smoother_damping(state))
upper_solve_arguments(state::DILUState, x) = (
    x, state.work, state.values, state.inverse_diagonal,
    state.rowptr, state.colval, smoother_damping(state))

function ilu_solve!(x, state::Union{ILU0State,DILUState}, b)
    ensure_smoother_work!(state, b)
    lower_kernel, upper_kernel = solve_kernels(state)
    lower! = lower_kernel(state.backend, state.block_size)
    upper! = upper_kernel(state.backend, state.block_size)
    launch_levels!(lower!, state.factor_offsets, state.factor_rows,
                   lower_solve_arguments(state, b)...)
    launch_levels!(upper!, state.upper_offsets, state.upper_rows,
                   upper_solve_arguments(state, x)...)
    x
end

function ilu_solve!(x::AbstractVector,
                    state::ILU0State{F,D,RP,CV},
                    b::AbstractVector) where {F,D,RP<:Vector,CV}
    ensure_smoother_work!(state, b)
    ilu_solve_cpu!(x, state, b, state.work)
end

function ilu_solve_cpu!(x, state::ILU0State, b, work)
    factors = state.factors
    inverse_diagonal = state.inverse_diagonal
    rowptr = state.rowptr
    colval = state.colval
    @inbounds for i in 1:state.n
        value = b[i]
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[k]
            j < i && (value -= factors[k] * work[j])
        end
        work[i] = value
    end
    damping = smoother_damping(state)
    @inbounds for i in state.n:-1:1
        value = work[i]
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[k]
            j > i && (value -= factors[k] * x[j])
        end
        x[i] = damping * (inverse_diagonal[i] * value)
    end
    x
end

function ilu_solve!(x::AbstractVector,
                    state::DILUState{D,AV,RP,CV},
                    b::AbstractVector) where {D,AV,RP<:Vector,CV}
    ensure_smoother_work!(state, b)
    ilu_solve_cpu!(x, state, b, state.work)
end

function ilu_solve_cpu!(x, state::DILUState, b, work)
    values = state.values
    inverse_diagonal = state.inverse_diagonal
    rowptr = state.rowptr
    colval = state.colval
    @inbounds for i in 1:state.n
        value = b[i]
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[k]
            j < i && (value -= values[k] * work[j])
        end
        work[i] = inverse_diagonal[i] * value
    end
    damping = smoother_damping(state)
    @inbounds for i in state.n:-1:1
        correction = zero(eltype(x))
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[k]
            j > i && (correction += values[k] * x[j])
        end
        x[i] = damping * (work[i] - inverse_diagonal[i] * correction)
    end
    x
end

smooth_kernels(::ILU0State) = (
    ilu0_smooth_lower_level_kernel!, ilu0_smooth_upper_level_kernel!)
smooth_kernels(::DILUState) = (
    dilu_smooth_lower_level_kernel!, dilu_smooth_upper_level_kernel!)
lower_smooth_arguments(state::ILU0State, A, x, b) = (
    state.work, b, x, A.nzval, state.factors, state.rowptr, state.colval)
lower_smooth_arguments(state::DILUState, A, x, b) = (
    state.work, b, x, state.values, state.inverse_diagonal,
    state.rowptr, state.colval)
upper_smooth_arguments(state::ILU0State, x) = (
    x, state.residual, state.work, state.factors, state.inverse_diagonal,
    state.rowptr, state.colval, smoother_damping(state))
upper_smooth_arguments(state::DILUState, x) = (
    x, state.residual, state.work, state.values, state.inverse_diagonal,
    state.rowptr, state.colval, smoother_damping(state))

function ilu_smooth_result!(x, A::StaticSparsityMatrixCSR, b,
                            state::Union{ILU0State,DILUState})
    ensure_smoother_work!(state, b)
    lower_kernel, upper_kernel = smooth_kernels(state)
    lower! = lower_kernel(state.backend, state.block_size)
    upper! = upper_kernel(state.backend, state.block_size)
    launch_levels!(lower!, state.factor_offsets, state.factor_rows,
                   lower_smooth_arguments(state, A, x, b)...)
    launch_levels!(upper!, state.upper_offsets, state.upper_rows,
                   upper_smooth_arguments(state, x)...)
    x
end

function ilu_smooth_result!(x::AbstractVector, A::StaticSparsityMatrixCSR,
                            b::AbstractVector,
                            state::ILU0State{F,D,RP,CV}) where {F,D,RP<:Vector,CV}
    ensure_smoother_work!(state, b)
    ilu_smooth_result_cpu!(x, A, b, state, state.work, state.residual)
end

function ilu_smooth_result_cpu!(x, A, b, state::ILU0State, work, correction)
    factors = state.factors
    inverse_diagonal = state.inverse_diagonal
    rowptr = state.rowptr
    colval = state.colval
    @inbounds for i in 1:state.n
        value = b[i]
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[k]
            value -= A.nzval[k] * x[j]
            j < i && (value -= factors[k] * work[j])
        end
        work[i] = value
    end
    damping = smoother_damping(state)
    @inbounds for i in state.n:-1:1
        value = work[i]
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[k]
            j > i && (value -= factors[k] * correction[j])
        end
        delta = damping * (inverse_diagonal[i] * value)
        correction[i] = delta
        x[i] += delta
    end
    x
end

function ilu_smooth_result!(x::AbstractVector, A::StaticSparsityMatrixCSR,
                            b::AbstractVector,
                            state::DILUState{D,AV,RP,CV}) where {D,AV,RP<:Vector,CV}
    ensure_smoother_work!(state, b)
    ilu_smooth_result_cpu!(x, A, b, state, state.work, state.residual)
end

function ilu_smooth_result_cpu!(x, A, b, state::DILUState, work, correction)
    values = state.values
    inverse_diagonal = state.inverse_diagonal
    rowptr = state.rowptr
    colval = state.colval
    @inbounds for i in 1:state.n
        value = b[i]
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[k]
            value -= A.nzval[k] * x[j]
            j < i && (value -= values[k] * work[j])
        end
        work[i] = inverse_diagonal[i] * value
    end
    damping = smoother_damping(state)
    @inbounds for i in state.n:-1:1
        value = work[i]
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[k]
            j > i && (value -= inverse_diagonal[i] * values[k] * correction[j])
        end
        delta = damping * value
        correction[i] = delta
        x[i] += delta
    end
    x
end

function apply!(x::AbstractVector, state::Union{ILU0State,DILUState},
                b::AbstractVector)
    require_apply_dimensions(x, state, b)
    same_backend(KernelAbstractions.get_backend(x), state.backend) ||
        throw(ArgumentError("output and smoother must use the same backend"))
    ilu_solve!(x, state, b)
end

function apply_correction!(x, state::Union{ILU0State,DILUState}, residual)
    ilu_solve!(state.residual, state, residual)
    axpy!(x, state.residual, one(smoother_damping(state)), state.backend,
           state.block_size)
    x
end

function smooth_level!(x, A::StaticSparsityMatrixCSR, b, state::Union{ILU0State,DILUState},
                  steps::Int; residual=nothing, zero_initial::Bool=false)
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
        smooth!(x, state, A, b; steps=steps - start + 1)
    end
    x
end

function smooth_result!(x, A::StaticSparsityMatrixCSR, b,
                         state::Union{ILU0State,DILUState}, steps::Int;
                         residual=nothing, zero_initial::Bool=false)
    steps > 0 || throw(ArgumentError("smoothing steps must be positive"))
    if isnothing(residual) && !zero_initial
        ilu_smooth_result!(x, A, b, state)
        if steps > 1
            smooth!(x, state, A, b; steps=steps - 1)
        end
        return x
    end
    smooth_level!(x, A, b, state, steps;
                  residual=residual, zero_initial=zero_initial)
end
