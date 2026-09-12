@inline function device_find_column(rowptr, colval, row, column)
    lo = rowptr[row]
    hi = rowptr[row + 1] - one(eltype(rowptr))
    while lo <= hi
        mid = (lo + hi) >>> 1
        value = colval[mid]
        if value == column
            return mid
        elseif value < column
            lo = mid + one(eltype(rowptr))
        else
            hi = mid - one(eltype(rowptr))
        end
    end
    zero(eltype(rowptr))
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

function grouped_schedule(groups::Vector{Int}, n::Int; reverse::Bool=false)
    number_of_groups = maximum(groups; init=0)
    counts = zeros(Int, number_of_groups)
    @inbounds for group in groups
        counts[group] += 1
    end
    ordered_counts = reverse ? Base.reverse(counts) : counts
    offsets = Vector{Int}(undef, number_of_groups + 1)
    offsets[1] = 1
    @inbounds for group in 1:number_of_groups
        offsets[group + 1] = offsets[group] + ordered_counts[group]
    end
    cursor = copy(offsets)
    rows = Vector{eltype(groups)}(undef, n)
    @inbounds for i in 1:n
        group = reverse ? number_of_groups + 1 - groups[i] : groups[i]
        rows[cursor[group]] = i
        cursor[group] += 1
    end
    offsets, rows
end

function greedy_row_coloring(rowptr::Vector{Ti}, colval::Vector{Ti}, n::Int) where Ti
    colors = zeros(Int, n)
    marks = zeros(Int, n + 1)
    @inbounds for i in 1:n
        for k in rowptr[i]:(rowptr[i + 1] - one(Ti))
            j = Int(colval[k])
            if j != i && 1 <= j <= n
                color = colors[j]
                !iszero(color) && (marks[color] = i)
            end
        end
        color = 1
        while marks[color] == i
            color += 1
        end
        colors[i] = color
    end
    lower_offsets, lower_rows = grouped_schedule(colors, n)
    upper_offsets, upper_rows = grouped_schedule(colors, n; reverse=true)
    colors, lower_offsets, lower_rows, upper_offsets, upper_rows
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
                                            @Const(ordering),
                                            @Const(rows), first, count)
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        original_diagonal = values[diagonal_positions[i]]
        diagonal = original_diagonal
        row_order = ordering[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            opposite = transpose_positions[k]
            if ordering[j] < row_order && !iszero(opposite)
                diagonal -= values[k] * inverse_diagonal[j] * values[opposite]
            end
        end
        inverse_diagonal[i] = robust_inverse(diagonal, original_diagonal)
    end
end

@inline function robust_inverse(diagonal::Number, original::Number)
    T = typeof(real(original))
    scale = max(abs(original), one(T))
    tolerance = sqrt(eps(T))*scale
    pivot = if isfinite(diagonal) && abs(diagonal) > tolerance
        diagonal
    elseif isfinite(original) && abs(original) > tolerance
        original
    else
        tolerance
    end
    inv(pivot)
end

@inline function finite_entries(value)
    finite = true
    @inbounds for entry in value
        finite &= isfinite(entry)
    end
    finite
end

@inline function robust_inverse(diagonal, original)
    candidate = inv(diagonal)
    finite_entries(candidate) && return candidate
    candidate = inv(original)
    finite_entries(candidate) && return candidate
    T = eltype(original)
    scale = one(T)
    @inbounds for entry in original
        scale = max(scale, abs(entry))
    end
    inv(original + sqrt(eps(T))*scale*one(original))
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
                                           @Const(ordering),
                                           @Const(rows), first, count)
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = rhs[i]
        row_order = ordering[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            ordering[j] < row_order && (value -= values[k] * work[j])
        end
        @inbounds work[i] = inverse_diagonal[i] * value
    end
end

@kernel function dilu_upper_level_kernel!(x, @Const(work), @Const(values),
                                           @Const(inverse_diagonal),
                                           @Const(rowptr), @Const(colval),
                                           @Const(ordering), damping,
                                           @Const(rows), first, count)
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        correction = zero(eltype(x))
        row_order = ordering[i]
        diagonal_inverse = inverse_diagonal[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            ordering[j] > row_order && (correction += values[k] * x[j])
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
        @Const(rowptr), @Const(colval), @Const(ordering),
        @Const(rows), first, count)
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = rhs[i]
        row_order = ordering[i]
        diagonal_inverse = inverse_diagonal[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            value -= values[k] * x[j]
            ordering[j] < row_order && (value -= values[k] * work[j])
        end
        @inbounds work[i] = diagonal_inverse * value
    end
end

@kernel function dilu_smooth_upper_level_kernel!(x, correction, @Const(work),
        @Const(values), @Const(inverse_diagonal), @Const(rowptr),
        @Const(colval), @Const(ordering), damping,
        @Const(rows), first, count)
    q = @index(Global)
    if q <= count
        i = rows[first + q - 1]
        value = work[i]
        row_order = ordering[i]
        diagonal_inverse = inverse_diagonal[i]
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - one(eltype(rowptr)))
            j = colval[k]
            ordering[j] > row_order &&
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

function setup_smoother(A::StaticSparsityMatrixCSR, config::ILU0; reuse=nothing)
    matrix_nrows(A) == matrix_ncols(A) || throw(DimensionMismatch("ILU0 requires a square matrix"))
    if reuse isa ILU0State && same_backend(reuse.backend, matrix_backend(A)) &&
       same_smoother_pattern(reuse, A)
        reuse.config = config
        return update_smoother!(reuse, A)
    end
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
                      matrix_backend(A), matrix_block_size(A), matrix_nrows(A))
    update_smoother!(state, A)
end

function setup_smoother(A::StaticSparsityMatrixCSR, config::DILU; reuse=nothing)
    matrix_nrows(A) == matrix_ncols(A) || throw(DimensionMismatch("DILU requires a square matrix"))
    if reuse isa DILUState && same_backend(reuse.backend, matrix_backend(A)) &&
       same_smoother_pattern(reuse, A)
        reuse.config = config
        return update_smoother!(reuse, A)
    end
    symbolic = ilu_symbolic(A)
    backend = matrix_backend(A)
    if backend isa KernelAbstractions.CPU
        ordering = collect(1:matrix_nrows(A))
        factor_offsets = symbolic.factor_offsets
        factor_rows = symbolic.factor_rows
        upper_offsets = symbolic.upper_offsets
        upper_rows = symbolic.upper_rows
    else
        ordering, factor_offsets, factor_rows, upper_offsets, upper_rows =
            greedy_row_coloring(symbolic.rowptr, symbolic.colval,
                                matrix_nrows(A))
    end
    values, inverse_diagonal = allocate_factor_storage(A)
    state = DILUState(inverse_diagonal, nothing, nothing, values,
                      A.rowptr, A.colval,
                      backend_copy(backend, symbolic.diagonal),
                      backend_copy(backend, symbolic.transpose),
                      backend_copy(backend, ordering),
                      factor_offsets,
                      backend_copy(backend, factor_rows),
                      upper_offsets,
                      backend_copy(backend, upper_rows),
                      symbolic.rowptr, symbolic.colval, config,
                      backend, matrix_block_size(A), matrix_nrows(A))
    update_smoother!(state, A)
end

setup_smoother(A::SparseMatrixCSC, config::Union{ILU0,DILU}; reuse=nothing) =
    setup_smoother(csr_matrix(A), config; reuse=reuse)

function update_smoother!(state::ILU0State, A::StaticSparsityMatrixCSR)
    require_same_smoother_pattern(state, A)
    copyto!(state.factors, 1, A.nzval, 1, matrix_nonzeros(A))
    kernel! = ilu0_factor_level_kernel!(state.backend, state.block_size)
    launch_levels!(kernel!, state.factor_offsets, state.factor_rows,
                    state.factors, state.inverse_diagonal, state.rowptr,
                    state.colval, state.diagonal_positions)
    state
end

function update_smoother!(state::DILUState, A::StaticSparsityMatrixCSR)
    require_same_smoother_pattern(state, A)
    copyto!(state.values, 1, A.nzval, 1, matrix_nonzeros(A))
    kernel! = dilu_factor_level_kernel!(state.backend, state.block_size)
    launch_levels!(kernel!, state.factor_offsets, state.factor_rows,
                    state.inverse_diagonal, state.values, state.rowptr,
                    state.colval, state.diagonal_positions,
                    state.transpose_positions, state.ordering)
    state
end

function ilu_solve!(x, state::ILU0State, b)
    ensure_smoother_work!(state, b)
    lower! = ilu0_lower_level_kernel!(state.backend, state.block_size)
    upper! = ilu0_upper_level_kernel!(state.backend, state.block_size)
    launch_levels!(lower!, state.factor_offsets, state.factor_rows,
                    state.work, b, state.factors, state.rowptr, state.colval)
    launch_levels!(upper!, state.upper_offsets, state.upper_rows,
                    x, state.work, state.factors, state.inverse_diagonal,
                    state.rowptr, state.colval, state.config.damping)
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
    damping = state.config.damping
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

function ilu_solve!(x, state::DILUState, b)
    ensure_smoother_work!(state, b)
    lower! = dilu_lower_level_kernel!(state.backend, state.block_size)
    upper! = dilu_upper_level_kernel!(state.backend, state.block_size)
    launch_levels!(lower!, state.factor_offsets, state.factor_rows,
                   state.work, b, state.values, state.inverse_diagonal,
                   state.rowptr, state.colval, state.ordering)
    launch_levels!(upper!, state.upper_offsets, state.upper_rows,
                   x, state.work, state.values, state.inverse_diagonal,
                   state.rowptr, state.colval, state.ordering,
                   state.config.damping)
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
    damping = state.config.damping
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

function ilu_smooth_result!(x, A::StaticSparsityMatrixCSR,
                            b, state::ILU0State)
    ensure_smoother_work!(state, b)
    lower! = ilu0_smooth_lower_level_kernel!(state.backend, state.block_size)
    upper! = ilu0_smooth_upper_level_kernel!(state.backend, state.block_size)
    launch_levels!(lower!, state.factor_offsets, state.factor_rows,
                   state.work, b, x, A.nzval, state.factors,
                   state.rowptr, state.colval)
    launch_levels!(upper!, state.upper_offsets, state.upper_rows,
                   x, state.residual, state.work, state.factors,
                   state.inverse_diagonal, state.rowptr, state.colval,
                   state.config.damping)
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
    damping = state.config.damping
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

function ilu_smooth_result!(x, A::StaticSparsityMatrixCSR,
                            b, state::DILUState)
    ensure_smoother_work!(state, b)
    lower! = dilu_smooth_lower_level_kernel!(state.backend, state.block_size)
    upper! = dilu_smooth_upper_level_kernel!(state.backend, state.block_size)
    launch_levels!(lower!, state.factor_offsets, state.factor_rows,
                   state.work, b, x, state.values, state.inverse_diagonal,
                   state.rowptr, state.colval, state.ordering)
    launch_levels!(upper!, state.upper_offsets, state.upper_rows,
                   x, state.residual, state.work, state.values,
                   state.inverse_diagonal, state.rowptr, state.colval,
                   state.ordering, state.config.damping)
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
    damping = state.config.damping
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
    length(x) == state.n || throw(DimensionMismatch())
    length(b) == state.n || throw(DimensionMismatch())
    same_backend(KernelAbstractions.get_backend(x), state.backend) ||
        throw(ArgumentError("output and smoother must use the same backend"))
    ilu_solve!(x, state, b)
end

function apply_correction!(x, state::Union{ILU0State,DILUState}, residual)
    ilu_solve!(state.residual, state, residual)
    axpy!(x, state.residual, one(state.config.damping), state.backend,
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

function update_level_smoother!(state::Union{ILU0State,DILUState}, A::StaticSparsityMatrixCSR,
                           options::AMGOptions)
    state.config = options.smoother
    update_smoother!(state, A)
end
