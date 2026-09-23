optional_property(::Nothing, field::Symbol) = nothing
optional_property(value, field::Symbol) = getproperty(value, field)

host_copy_source(src::BitVector) = Vector{Bool}(src)
host_copy_source(src) = src

function host_buffer(old, ::Type{T}, n::Integer; zeroed::Bool = false) where {T}
    if old isa Vector{T}
        resize!(old, Int(n))
        zeroed && fill!(old, zero(T))
        return old
    end
    return if zeroed
        zeros(T, Int(n))
    else
        Vector{T}(undef, Int(n))
    end
end

function strength(
        A::StaticSparsityMatrixCSR{Tv, Ti}, theta::Real, max_row_sum::Real = 1.0,
        reuse = nothing
    ) where {Tv, Ti}
    backend = matrix_backend(A)
    scalar_type = matrix_real_type(Tv)
    theta = convert(scalar_type, theta)
    max_row_sum = convert(scalar_type, max_row_sum)
    number_of_nonzeros = matrix_nonzeros(A)
    n = matrix_nrows(A)
    if reuse isa AbstractVector{Bool} && buffer_backend_matches(reuse, backend) &&
            (reuse isa Vector || length(reuse) == number_of_nonzeros)
        strong = if reuse isa Vector
            host_buffer(reuse, Bool, number_of_nonzeros)
        else
            reuse
        end
    else
        strong = backend_zeros(backend, Bool, number_of_nonzeros)
    end
    k! = strength_kernel!(backend, matrix_kernel_block_size(A))
    k!(
        strong, A.rowptr, A.colval, A.nzval, theta, max_row_sum, n;
        ndrange = n
    )
    synchronize_backend(backend)
    return strong
end

function strong_transpose(A::StaticSparsityMatrixCSR{Tv, Ti}, strong, workspace = nothing) where {Tv, Ti}
    counts = host_buffer(
        optional_property(workspace, :ti1),
        Ti, matrix_nrows(A); zeroed = true
    )
    @inbounds for i in 1:matrix_nrows(A), k in nzrange(A, i)
        strong[k] && (counts[A.colval[k]] += one(Ti))
    end
    offsets = host_buffer(
        optional_property(workspace, :ti2),
        Ti, matrix_nrows(A) + 1
    )
    offsets[1] = one(Ti)
    @inbounds for i in 1:matrix_nrows(A)
        offsets[i + 1] = offsets[i] + counts[i]
    end
    cursor = counts
    copyto!(cursor, 1, offsets, 1, matrix_nrows(A))
    sources = host_buffer(
        optional_property(workspace, :ti3),
        Ti, Int(offsets[end] - 1)
    )
    @inbounds for i in 1:matrix_nrows(A), k in nzrange(A, i)
        if strong[k]
            j = A.colval[k]
            sources[cursor[j]] = Ti(i)
            cursor[j] += one(Ti)
        end
    end
    return offsets, sources
end

function degree_order(measure, ::Type{Ti}) where {Ti}
    n = length(measure)
    max_degree = if isempty(measure)
        0
    else
        maximum(measure)
    end
    bucket_counts = zeros(Int, Int(max_degree) + 1)
    @inbounds for degree in measure
        bucket_counts[Int(degree) + 1] += 1
    end
    bucket_starts = similar(bucket_counts)
    next_position = 1
    @inbounds for degree in Int(max_degree):-1:0
        bucket_starts[degree + 1] = next_position
        next_position += bucket_counts[degree + 1]
    end
    cursor = copy(bucket_starts)
    order = Vector{Ti}(undef, n)
    # Ascending input order preserves the index tie-break within each bucket.
    @inbounds for i in 1:n
        bucket = Int(measure[i]) + 1
        order[cursor[bucket]] = Ti(i)
        cursor[bucket] += 1
    end
    return order
end

function aggregation_split(
        A::StaticSparsityMatrixCSR{Tv, Ti}, strong, reuse_cf = nothing,
        reuse_map = nothing, workspace = nothing
    ) where {Tv, Ti}
    n = matrix_nrows(A)
    agg = zeros(Int, n)
    seed = falses(n)
    nagg = 0
    degree = zeros(Int, n)
    @inbounds for i in 1:n, k in nzrange(A, i)
        strong[k] && (degree[i] += 1)
    end
    offsets, sources = strong_transpose(A, strong, workspace)
    order = degree_order(degree, Ti)
    @inbounds for i in order
        agg[i] != 0 && continue
        nagg += 1
        agg[i] = nagg
        seed[i] = true
        for k in nzrange(A, i)
            j = A.colval[k]
            strong[k] && agg[j] == 0 && (agg[j] = nagg)
        end
        # Include transpose neighbors as well; reservoir matrices need not be symmetric.
        for q in offsets[i]:(offsets[i + 1] - 1)
            j = sources[q]
            agg[j] == 0 && (agg[j] = nagg)
        end
    end
    cf = host_buffer(reuse_cf, Int8, n)
    fill!(cf, Int8(-1))
    coarse_map = host_buffer(reuse_map, Ti, n; zeroed = true)
    @inbounds for i in 1:n
        cf[i] = if seed[i]
            Int8(1)
        else
            Int8(-1)
        end
        coarse_map[i] = agg[i]
    end
    return cf, coarse_map, nagg
end

function cf_split(
        A::StaticSparsityMatrixCSR{Tv, Ti}, strong, ::AbstractCoarsening,
        reuse_cf = nothing, reuse_map = nothing, workspace = nothing
    ) where {Tv, Ti}
    n = matrix_nrows(A)
    cf = host_buffer(reuse_cf, Int8, n; zeroed = true)
    offsets, sources = strong_transpose(A, strong, workspace)
    measure = Vector{Ti}(undef, n)
    @inbounds for i in 1:n
        measure[i] = offsets[i + 1] - offsets[i]
    end
    order = degree_order(measure, Ti)
    @inbounds for best in order
        cf[best] == 0 || continue
        cf[best] = 1
        # Strong neighbors in either direction become F points.
        for k in nzrange(A, best)
            j = A.colval[k]
            if strong[k] && cf[j] == 0
                cf[j] = -1
            end
        end
        for q in offsets[best]:(offsets[best + 1] - 1)
            j = sources[q]
            cf[j] == 0 && (cf[j] = -1)
        end
    end
    coarse_map = host_buffer(reuse_map, Ti, n; zeroed = true)
    nc = 0
    @inbounds for i in 1:n
        if cf[i] == 1
            nc += 1
            coarse_map[i] = nc
        end
    end
    return cf, coarse_map, nc
end

@inline function bucket_remove!(
        i::Int, measure::Vector{Int}, head::Vector{Int},
        tail::Vector{Int}, next::Vector{Int}, prev::Vector{Int}
    )
    @inbounds begin
        bucket = measure[i] + 1
        p, q = prev[i], next[i]
        if p == 0
            head[bucket] = q
        else
            next[p] = q
        end
        if q == 0
            tail[bucket] = p
        else
            prev[q] = p
        end
        next[i] = 0
        prev[i] = 0
    end
    return nothing
end

@inline function bucket_move!(
        i::Int, value::Int, measure::Vector{Int},
        head::Vector{Int}, tail::Vector{Int},
        next::Vector{Int}, prev::Vector{Int}
    )
    bucket_remove!(i, measure, head, tail, next, prev)
    bucket = value + 1
    if bucket > length(head)
        old = length(head)
        resize!(head, bucket)
        resize!(tail, bucket)
        fill!(view(head, (old + 1):bucket), 0)
        fill!(view(tail, (old + 1):bucket), 0)
    end
    @inbounds begin
        measure[i] = value
        p = tail[bucket]
        tail[bucket] = i
        prev[i] = p
        next[i] = 0
        if p == 0
            head[bucket] = i
        else
            next[p] = i
        end
    end
    return nothing
end


"""HYPRE-compatible RS first pass used by HMIS."""
function hmis_rs_first_pass!(
        A::StaticSparsityMatrixCSR, strong, offsets, sources, cf,
        workspace = nothing
    )
    n = matrix_nrows(A)
    measure = host_buffer(optional_property(workspace, :int1), Int, n)
    @inbounds for i in 1:n
        measure[i] = offsets[i + 1] - offsets[i]
    end

    # HYPRE uses Z_PT (-2) for zero-measure points in the HMIS first pass.
    @inbounds for i in 1:n
        has_out = false
        for k in nzrange(A, i)
            if strong[k]
                has_out = true
                break
            end
        end
        !has_out && measure[i] == 0 && (cf[i] = Int8(-2))
    end
    @inbounds for i in 1:n
        cf[i] == 0 || continue
        if measure[i] == 0
            cf[i] = Int8(-2)
            for k in nzrange(A, i)
                j = Int(A.colval[k])
                strong[k] && cf[j] == 0 && (measure[j] += 1)
            end
        end
    end

    top = 0
    @inbounds for i in 1:n
        cf[i] == 0 && (top = max(top, measure[i]))
    end
    head = host_buffer(
        optional_property(workspace, :int2),
        Int, top + 1; zeroed = true
    )
    tail = host_buffer(
        optional_property(workspace, :int3),
        Int, top + 1; zeroed = true
    )
    next = host_buffer(
        optional_property(workspace, :int4),
        Int, n; zeroed = true
    )
    prev = host_buffer(
        optional_property(workspace, :int5),
        Int, n; zeroed = true
    )
    @inbounds for i in 1:n
        cf[i] == 0 || continue
        bucket = measure[i] + 1
        p = tail[bucket]
        tail[bucket] = i
        prev[i] = p
        if p == 0
            head[bucket] = i
        else
            next[p] = i
        end
    end

    while true
        while top >= 0 && head[top + 1] == 0
            top -= 1
        end
        top < 0 && break
        best = head[top + 1]
        bucket_remove!(best, measure, head, tail, next, prev)
        cf[best] = Int8(1)

        # Points strongly depending on the new C point become F points.
        @inbounds for q in offsets[best]:(offsets[best + 1] - 1)
            j = Int(sources[q])
            cf[j] == 0 || continue
            bucket_remove!(j, measure, head, tail, next, prev)
            cf[j] = Int8(-1)
            for k in nzrange(A, j)
                l = Int(A.colval[k])
                if strong[k] && cf[l] == 0
                    value = measure[l] + 1
                    bucket_move!(l, value, measure, head, tail, next, prev)
                    top = max(top, value)
                end
            end
        end

        # Remove the selected point's influence from its outgoing neighbors.
        @inbounds for k in nzrange(A, best)
            j = Int(A.colval[k])
            strong[k] && cf[j] == 0 || continue
            value = max(measure[j] - 1, 0)
            bucket_move!(j, value, measure, head, tail, next, prev)
            if value == 0
                bucket_remove!(j, measure, head, tail, next, prev)
                cf[j] = Int8(-2)
                for q in nzrange(A, j)
                    l = Int(A.colval[q])
                    if strong[q] && cf[l] == 0
                        value_l = measure[l] + 1
                        bucket_move!(l, value_l, measure, head, tail, next, prev)
                        top = max(top, value_l)
                    end
                end
            end
        end
    end
    return cf
end

function hypre_randomized_measure(offsets, n::Int, workspace = nothing)
    measure = host_buffer(
        optional_property(workspace, :real1),
        Float64, n
    )
    seed = Int64(2747)
    inv_modulus = 1.0 / 2147483647.0
    @inbounds for i in 1:n
        high, low = div(seed, 127773), seed % 127773
        candidate = 16807low - 2836high
        seed = if candidate > 0
            candidate
        else
            candidate + 2147483647
        end
        measure[i] = Float64(offsets[i + 1] - offsets[i]) + seed * inv_modulus
    end
    return measure
end

function hmis_pmis_finish!(A::StaticSparsityMatrixCSR, strong, cf, measure)
    n = matrix_nrows(A)
    @inbounds for i in 1:n
        if cf[i] == -1
            cf[i] = 0
        elseif cf[i] == -2
            has_out = false
            for k in nzrange(A, i)
                if strong[k]
                    has_out = true
                    break
                end
            end
            cf[i] = if measure[i] >= 1.0 || has_out
                Int8(0)
            else
                Int8(-1)
            end
        end
    end

    iteration = 1
    while true
        any_undecided = false
        if iteration > 1
            @inbounds for i in 1:n
                cf[i] == 0 || continue
                any_undecided = true
                if measure[i] < 1.0
                    cf[i] = Int8(-1)
                    continue
                end
                local_maximum = true
                for k in nzrange(A, i)
                    j = Int(A.colval[k])
                    if strong[k] && cf[j] != -1 && measure[j] > measure[i]
                        local_maximum = false
                        break
                    end
                end
                local_maximum && (cf[i] = Int8(2))
            end
            @inbounds for i in 1:n
                cf[i] == 2 || continue
                cf[i] = Int8(1)
                for k in nzrange(A, i)
                    j = Int(A.colval[k])
                    strong[k] && cf[j] == 0 && (measure[j] -= 1.0)
                end
            end
        else
            @inbounds for i in 1:n
                cf[i] == 0 || continue
                any_undecided = true
                measure[i] < 1.0 && (cf[i] = Int8(-1))
            end
        end

        @inbounds for i in 1:n
            cf[i] == 0 || continue
            for k in nzrange(A, i)
                if strong[k] && cf[Int(A.colval[k])] == 1
                    cf[i] = Int8(-1)
                    break
                end
            end
        end
        @inbounds for i in 1:n
            cf[i] == 0 || (measure[i] = 0.0)
        end
        any_undecided || break
        iteration += 1
        iteration <= n + 1 || break
    end
    @inbounds for i in 1:n
        cf[i] == 0 && (cf[i] = Int8(-1))
    end
    return cf
end

function cf_split(
        A::StaticSparsityMatrixCSR{Tv, Ti}, strong, ::HMIS,
        reuse_cf = nothing, reuse_map = nothing, workspace = nothing
    ) where {Tv, Ti}
    n = matrix_nrows(A)
    cf = host_buffer(reuse_cf, Int8, n; zeroed = true)
    offsets, sources = strong_transpose(A, strong, workspace)
    hmis_rs_first_pass!(A, strong, offsets, sources, cf, workspace)
    measure = hypre_randomized_measure(offsets, n, workspace)
    hmis_pmis_finish!(A, strong, cf, measure)

    coarse_map = host_buffer(reuse_map, Ti, n; zeroed = true)
    nc = 0
    @inbounds for i in 1:n
        if cf[i] == 1
            nc += 1
            coarse_map[i] = Ti(nc)
        end
    end
    return cf, coarse_map, nc
end

@inline function accumulate_candidate!(
        cols::Vector{Ti}, vals::Vector{Tv},
        col::Ti, value::Tv
    ) where {Tv, Ti}
    position = candidate_index(cols, col)
    if iszero(position)
        push!(cols, col)
        push!(vals, value)
    else
        @inbounds vals[position] += value
    end
    return nothing
end

@inline function candidate_index(cols, col)
    @inbounds for q in eachindex(cols)
        cols[q] == col && return q
    end
    return 0
end

@inline candidate_present(cols, col) = !iszero(candidate_index(cols, col))

function scale_candidate_weights!(
        values::Vector{Tv},
        effective_diagonal
    ) where {Tv}
    if abs(effective_diagonal) > eps(real(Tv))
        scale = -inv(effective_diagonal)
        @inbounds for q in eachindex(values)
            values[q] *= scale
        end
    end
    return nothing
end

function candidate_weights!(
        cols::Vector{Ti}, vals::Vector{Tv},
        A::StaticSparsityMatrixCSR{Tv, Ti}, i, cf, cmap, strong,
        diagonal, ::ExtendedIInterpolation
    ) where {Tv, Ti}
    empty!(cols)
    empty!(vals)
    # Build HYPRE's C-hat set: strong direct C neighbors and C points reached
    # through a strong F neighbor. The encounter order is deterministic.
    @inbounds for k in nzrange(A, i)
        j = A.colval[k]
        strong[k] || continue
        if cf[j] == 1
            accumulate_candidate!(cols, vals, cmap[j], zero(Tv))
        elseif cf[j] == -1
            for q in nzrange(A, j)
                l = A.colval[q]
                if cf[l] == 1 && strong[q]
                    accumulate_candidate!(cols, vals, cmap[l], zero(Tv))
                end
            end
        end
    end
    isempty(cols) && return nothing

    # Extended+i / ExtPI weights. Weak connections and connections outside
    # C-hat are lumped into the effective diagonal. Strong F contributions are
    # distributed over C-hat using the sign-compatible part of that F row.
    effective_diagonal = zero(Tv)
    @inbounds for k in nzrange(A, i)
        j = A.colval[k]
        aij = A.nzval[k]
        if j == i
            effective_diagonal += aij
            continue
        end

        direct = cf[j] == 1 ? candidate_index(cols, cmap[j]) : 0
        if direct != 0
            vals[direct] += aij
        elseif strong[k] && cf[j] == -1
            diag_j = diagonal[j]
            sign_j = if real(diag_j) < 0
                -1
            else
                1
            end
            denominator = zero(Tv)
            for q in nzrange(A, j)
                l = A.colval[q]
                l == j && continue
                ajl = A.nzval[q]
                sign_j * real(ajl) < 0 || continue
                included = l == i
                if !included && cf[l] == 1
                    included = candidate_present(cols, cmap[l])
                end
                included && (denominator += ajl)
            end
            if abs(denominator) > eps(real(Tv))
                distribute = aij / denominator
                for q in nzrange(A, j)
                    l = A.colval[q]
                    l == j && continue
                    ajl = A.nzval[q]
                    sign_j * real(ajl) < 0 || continue
                    if l == i
                        effective_diagonal += distribute * ajl
                    elseif cf[l] == 1
                        position = candidate_index(cols, cmap[l])
                        iszero(position) || (vals[position] += distribute * ajl)
                    end
                end
            else
                effective_diagonal += aij
            end
        else
            effective_diagonal += aij
        end
    end
    return scale_candidate_weights!(vals, effective_diagonal)
end

function candidate_weights!(
        cols::Vector{Ti}, vals::Vector{Tv},
        A::StaticSparsityMatrixCSR{Tv, Ti}, i, cf, cmap, strong,
        diagonal, ::ClassicalInterpolation
    ) where {Tv, Ti}
    empty!(cols)
    empty!(vals)
    effective_diagonal = diagonal[i]

    # Classical interpolation starts from direct strong C-neighbors. Weak
    # connections are lumped into the diagonal.
    @inbounds for k in nzrange(A, i)
        j = A.colval[k]
        j == i && continue
        if strong[k] && cf[j] == 1
            accumulate_candidate!(cols, vals, cmap[j], A.nzval[k])
        elseif !strong[k]
            effective_diagonal += A.nzval[k]
        end
    end
    isempty(cols) && return nothing

    # Distribute each strong F-neighbor through the direct coarse set C_i.
    @inbounds for k in nzrange(A, i)
        j = A.colval[k]
        strong[k] && cf[j] == -1 || continue
        denominator = zero(Tv)
        for q in nzrange(A, j)
            l = A.colval[q]
            cf[l] == 1 || continue
            candidate_present(cols, cmap[l]) && (denominator += A.nzval[q])
        end
        if abs(denominator) > eps(real(Tv))
            distribute = A.nzval[k] / denominator
            for q in nzrange(A, j)
                l = A.colval[q]
                cf[l] == 1 || continue
                position = candidate_index(cols, cmap[l])
                iszero(position) || (vals[position] += distribute * A.nzval[q])
            end
        else
            effective_diagonal += A.nzval[k]
        end
    end

    return scale_candidate_weights!(vals, effective_diagonal)
end

@inline function candidate_score(value, p::Int)
    return if p == 2
        abs2(value)
    else
        abs(value)^p
    end
end

function sort_candidates!(cols, vals, p::Int)
    # Distance-two candidate sets are small; insertion sort avoids allocating a
    # Pair vector and its sorting workspace for every fine row.
    @inbounds for q in 2:length(cols)
        col, value = cols[q], vals[q]
        score = candidate_score(value, p)
        pos = q
        while pos > 1
            previous = candidate_score(vals[pos - 1], p)
            (score > previous || (score == previous && col < cols[pos - 1])) || break
            cols[pos], vals[pos] = cols[pos - 1], vals[pos - 1]
            pos -= 1
        end
        cols[pos], vals[pos] = col, value
    end
    return nothing
end

function sort_selected_columns!(cols, vals, count::Int)
    @inbounds for q in 2:count
        col, value = cols[q], vals[q]
        pos = q
        while pos > 1 && cols[pos - 1] > col
            cols[pos], vals[pos] = cols[pos - 1], vals[pos - 1]
            pos -= 1
        end
        cols[pos], vals[pos] = col, value
    end
    return nothing
end

@inline function candidate_count(vals, interpolation)
    isempty(vals) && return 0
    count = if iszero(interpolation.max_elements)
        length(vals)
    else
        min(length(vals), interpolation.max_elements)
    end
    if interpolation.truncation > 0
        leading_score = candidate_score(vals[1], interpolation.norm_p)
        truncation = convert(typeof(leading_score), interpolation.truncation)
        cutoff = truncation * leading_score
        @inbounds for q in 1:count
            if candidate_score(vals[q], interpolation.norm_p) < cutoff
                return q - 1
            end
        end
    end
    return count
end

function select_interpolation_candidates!(
        cols, vals, A, row, cf, cmap,
        strong, diagonal, interpolation
    )
    candidate_weights!(
        cols, vals, A, row, cf, cmap, strong, diagonal,
        interpolation
    )
    sort_candidates!(cols, vals, interpolation.norm_p)
    return candidate_count(vals, interpolation)
end

function with_setup_rows(f, n::Int)
    if Threads.nthreads() > 1 && n >= 4_096
        Threads.@threads :static for i in 1:n
            f(i, Threads.threadid())
        end
    else
        @inbounds for i in 1:n
            f(i, 1)
        end
    end
    return nothing
end

function same_prefix(a::AbstractVector, b::AbstractVector, n::Int)
    @inbounds for k in 1:n
        a[k] == b[k] || return false
    end
    return true
end

function same_csr_pattern(
        a::StaticSparsityMatrixCSR{Tv, Ti, <:Vector, <:Vector, <:Vector},
        b::StaticSparsityMatrixCSR{Tv, Ti}, workspace
    ) where {Tv, Ti}
    size(a) == size(b) && matrix_nonzeros(a) == matrix_nonzeros(b) || return false
    old_rp = host_prefix_reusing(workspace.ti2, b.rowptr, matrix_nrows(b) + 1)
    same_prefix(a.rowptr, old_rp, matrix_nrows(a) + 1) || return false
    old_cv = host_prefix_reusing(workspace.ti3, b.colval, matrix_nonzeros(b))
    return same_prefix(a.colval, old_cv, matrix_nonzeros(a))
end

function build_prolongation(
        A::StaticSparsityMatrixCSR{Tv, Ti}, cf, cmap, nc, strong,
        interpolation::AbstractInterpolation,
        reuse = nothing, workspace = nothing
    ) where {Tv, Ti}
    n = matrix_nrows(A)
    diagonal = host_buffer(
        optional_property(workspace, :values),
        Tv, n; zeroed = true
    )
    with_setup_rows(n) do i, _
        value = zero(Tv)
        @inbounds for k in nzrange(A, i)
            A.colval[k] == i && (value += A.nzval[k])
        end
        @inbounds diagonal[i] = value
    end

    nscratch = Threads.maxthreadid()
    scratch_cols = if isnothing(workspace)
        [sizehint!(Ti[], 32) for _ in 1:nscratch]
    else
        workspace.interpolation_cols
    end
    scratch_vals = if isnothing(workspace)
        [sizehint!(Tv[], 32) for _ in 1:nscratch]
    else
        workspace.interpolation_vals
    end
    counts = host_buffer(optional_property(workspace, :ti1), Ti, n)
    with_setup_rows(n) do i, tid
        if cf[i] == 1
            @inbounds counts[i] = one(Ti)
        else
            candidate_cols, candidate_vals = scratch_cols[tid], scratch_vals[tid]
            selected = select_interpolation_candidates!(
                candidate_cols, candidate_vals, A, i, cf, cmap, strong,
                diagonal, interpolation
            )
            @inbounds counts[i] = Ti(selected)
        end
    end

    old_rp = optional_property(reuse, :rowptr)
    old_cv = optional_property(reuse, :colval)
    old_pv = optional_property(reuse, :nzval)
    rp = host_buffer(old_rp, Ti, n + 1)
    rp[1] = one(Ti)
    @inbounds for i in 1:n
        rp[i + 1] = rp[i] + counts[i]
    end
    cols = host_buffer(old_cv, Ti, Int(rp[end] - one(Ti)))
    vals = host_buffer(old_pv, Tv, length(cols))
    with_setup_rows(n) do i, tid
        if cf[i] == 1
            @inbounds begin
                cols[rp[i]] = cmap[i]
                vals[rp[i]] = one(Tv)
            end
        else
            candidate_cols, candidate_vals = scratch_cols[tid], scratch_vals[tid]
            count = select_interpolation_candidates!(
                candidate_cols, candidate_vals, A, i, cf, cmap, strong,
                diagonal, interpolation
            )
            if count > 0
                scale = one(Tv)
                if interpolation.rescale && count < length(candidate_vals)
                    original_sum = zero(Tv)
                    kept_sum = zero(Tv)
                    @inbounds for q in eachindex(candidate_vals)
                        value = candidate_vals[q]
                        original_sum += value
                        q <= count && (kept_sum += value)
                    end
                    # Match HYPRE's interpolation truncation: retain the
                    # action of the untruncated row on a constant vector.
                    # Normalizing every row to one changes valid Ext+i
                    # weights even when no coefficient was discarded.
                    abs(kept_sum) > eps(real(Tv)) &&
                        (scale = original_sum / kept_sum)
                end
                sort_selected_columns!(candidate_cols, candidate_vals, count)
                @inbounds for q in 1:count
                    k = rp[i] + q - 1
                    cols[k] = candidate_cols[q]
                    vals[k] = candidate_vals[q] * scale
                end
            end
        end
    end
    return Prolongation{Tv, Ti, typeof(rp), typeof(cols), typeof(vals)}(rp, cols, vals, n, nc)
end

function build_aggregation_prolongation(
        ::Type{Tv}, ::Type{Ti}, cmap, n, nc,
        reuse = nothing
    ) where {Tv, Ti}
    old_rp = optional_property(reuse, :rowptr)
    old_cv = optional_property(reuse, :colval)
    old_pv = optional_property(reuse, :nzval)
    rp = host_buffer(old_rp, Ti, n + 1)
    @inbounds for i in 1:(n + 1)
        rp[i] = Ti(i)
    end
    cv = host_buffer(old_cv, Ti, n)
    copyto!(cv, cmap)
    pv = host_buffer(old_pv, Tv, n)
    fill!(pv, one(Tv))
    return Prolongation{Tv, Ti, typeof(rp), typeof(cv), typeof(pv)}(rp, cv, pv, n, nc)
end

function build_prolongation(
        A::StaticSparsityMatrixCSR{Tv, Ti}, cf, cmap, nc, strong,
        ::ConstantInterpolation,
        reuse = nothing, workspace = nothing
    ) where {Tv, Ti}
    return build_aggregation_prolongation(Tv, Ti, cmap, matrix_nrows(A), nc, reuse)
end

function transpose_map(
        P::Prolongation{Tv, Ti}, reuse = nothing,
        workspace = nothing
    ) where {Tv, Ti}
    counts = host_buffer(
        optional_property(workspace, :ti1),
        Ti, P.ncol; zeroed = true
    )
    @inbounds for i in 1:P.nrow, k in P.rowptr[i]:(P.rowptr[i + 1] - 1)
        counts[P.colval[k]] += one(Ti)
    end
    old_offsets = optional_property(reuse, :offsets)
    old_rows = optional_property(reuse, :fine_rows)
    old_indices = optional_property(reuse, :p_indices)
    offsets = host_buffer(old_offsets, Ti, P.ncol + 1)
    offsets[1] = one(Ti)
    @inbounds for J in 1:P.ncol
        offsets[J + 1] = offsets[J] + counts[J]
    end
    cursor = host_buffer(
        optional_property(workspace, :ti2),
        Ti, P.ncol + 1
    )
    copyto!(cursor, offsets)
    pnnz = Int(P.rowptr[P.nrow + 1] - one(Ti))
    rows = host_buffer(old_rows, Ti, pnnz)
    indices = host_buffer(old_indices, Ti, pnnz)
    @inbounds for i in 1:P.nrow, k in P.rowptr[i]:(P.rowptr[i + 1] - 1)
        J = P.colval[k]
        q = cursor[J]
        rows[q], indices[q] = Ti(i), Ti(k)
        cursor[J] += one(Ti)
    end
    return TransposeMap{Ti, typeof(offsets), typeof(rows), typeof(indices)}(offsets, rows, indices)
end

@inline function find_column(A::StaticSparsityMatrixCSR{Tv, Ti}, row::Ti, col::Ti) where {Tv, Ti}
    lo = A.rowptr[row]
    hi = A.rowptr[row + 1] - one(Ti)
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        value = A.colval[mid]
        if value < col
            lo = mid + one(Ti)
        else
            hi = mid - one(Ti)
        end
    end
    @boundscheck (lo < A.rowptr[row + 1] && A.colval[lo] == col) ||
        error("Galerkin pattern is missing ($row, $col)")
    return lo
end

function collect_galerkin_columns!(
        columns, markers, A, P, Pt,
        coarse_row, thread_id
    )
    empty!(columns)
    @inbounds for q in Pt.offsets[coarse_row]:(Pt.offsets[coarse_row + 1] - 1)
        fine_row = Pt.fine_rows[q]
        for a_index in nzrange(A, fine_row)
            fine_column = A.colval[a_index]
            for p_index in P.rowptr[fine_column]:(P.rowptr[fine_column + 1] - 1)
                coarse_column = P.colval[p_index]
                if markers[coarse_column, thread_id] != coarse_row
                    markers[coarse_column, thread_id] = coarse_row
                    push!(columns, coarse_column)
                end
            end
        end
    end
    return columns
end

function galerkin_structure(
        A::StaticSparsityMatrixCSR{Tv, Ti}, P::Prolongation{Tv, Ti},
        Pt::TransposeMap{Ti}, reuse = nothing,
        reuse_coarse = nothing, workspace = nothing
    ) where {Tv, Ti}
    nc = P.ncol
    parallel = Threads.nthreads() > 1 && nc >= 4_096
    nscratch = if parallel
        Threads.maxthreadid()
    else
        1
    end
    marker_count = nc * nscratch
    marker_storage = if isnothing(workspace)
        Vector{Ti}(undef, marker_count)
    elseif CAN_RESIZE_SHARED_ARRAY
        resize!(workspace.markers, marker_count)
    elseif length(workspace.markers) == marker_count
        workspace.markers
    else
        # On Julia 1.10, reshape marks the backing Array as shared and it
        # cannot subsequently be resized. Replace it when the size changes.
        workspace.markers = Vector{Ti}(undef, marker_count)
    end
    fill!(marker_storage, zero(Ti))
    markers = reshape(marker_storage, nc, nscratch)
    scratch = if isnothing(workspace)
        [sizehint!(Ti[], 64) for _ in 1:Threads.maxthreadid()]
    else
        workspace.galerkin_cols
    end
    row_counts = host_buffer(optional_property(workspace, :ti1), Ti, nc)

    # Discover the exact sorted coarse pattern in parallel, one coarse row at a
    # time. Dense thread-local markers make duplicate removal O(1) per triple
    # while remaining much smaller than SparseArrays' AP/RAP temporaries.
    with_setup_rows(nc) do II, tid
        I = Ti(II)
        columns = scratch[tid]
        collect_galerkin_columns!(columns, markers, A, P, Pt, I, tid)
        @inbounds row_counts[I] = Ti(length(columns))
    end

    old_rowptr = optional_property(reuse_coarse, :rowptr)
    old_colval = optional_property(reuse_coarse, :colval)
    old_nzval = optional_property(reuse_coarse, :nzval)
    same_pattern = !isnothing(reuse_coarse) && matrix_nrows(reuse_coarse) == nc &&
        matrix_ncols(reuse_coarse) == nc
    if same_pattern
        expected = one(Ti)
        @inbounds for I in 1:nc
            if old_rowptr[I] != expected
                same_pattern = false
                break
            end
            expected += row_counts[I]
        end
        same_pattern = same_pattern && old_rowptr[nc + 1] == expected
    end
    rowptr = host_buffer(old_rowptr, Ti, nc + 1)
    rowptr[1] = one(Ti)
    @inbounds for I in 1:nc
        rowptr[I + 1] = rowptr[I] + row_counts[I]
    end
    coarse_nnz = Int(rowptr[end] - one(Ti))
    colval = host_buffer(old_colval, Ti, coarse_nnz)
    pattern_flags = if isnothing(workspace)
        ones(Int, Threads.maxthreadid())
    else
        workspace.int1
    end
    resize!(pattern_flags, Threads.maxthreadid())
    fill!(pattern_flags, 1)
    fill!(markers, zero(Ti))
    with_setup_rows(nc) do II, tid
        I = Ti(II)
        columns = scratch[tid]
        collect_galerkin_columns!(columns, markers, A, P, Pt, I, tid)
        # The entries are unique, so stability is irrelevant. Explicit
        # in-place QuickSort avoids radix-sort scratch allocation per row.
        sort!(columns; alg = QuickSort)
        if same_pattern
            start = Int(rowptr[I])
            @inbounds for q in eachindex(columns)
                if old_colval[start + q - 1] != columns[q]
                    pattern_flags[tid] = 0
                    break
                end
            end
        end
        copyto!(colval, Int(rowptr[I]), columns, 1, length(columns))
    end
    same_pattern = same_pattern && all(==(1), pattern_flags)

    nzval = host_buffer(old_nzval, Tv, coarse_nnz; zeroed = true)
    Ac = csr_matrix(
        rowptr, colval, nzval, nc, nc;
        block_size = matrix_batch_size(A)
    )
    counts = host_buffer(
        optional_property(workspace, :ti1),
        Ti, coarse_nnz; zeroed = true
    )
    with_setup_rows(nc) do II, tid
        I = Ti(II)
        @inbounds for dest in rowptr[I]:(rowptr[I + 1] - 1)
            markers[colval[dest], tid] = dest
        end
        @inbounds for q in Pt.offsets[I]:(Pt.offsets[I + 1] - 1)
            i = Pt.fine_rows[q]
            pleft = Pt.p_indices[q]
            left_value = conj(P.nzval[pleft])
            for aidx in nzrange(A, i)
                j = A.colval[aidx]
                left_a = left_value * A.nzval[aidx]
                for pright in P.rowptr[j]:(P.rowptr[j + 1] - 1)
                    dest = markers[P.colval[pright], tid]
                    counts[dest] += one(Ti)
                    nzval[dest] += left_a * P.nzval[pright]
                end
            end
        end
    end

    old_offsets = optional_property(reuse, :offsets)
    old_left = optional_property(reuse, :p_left)
    old_a = optional_property(reuse, :a_index)
    old_right = optional_property(reuse, :p_right)
    offsets = host_buffer(old_offsets, Ti, coarse_nnz + 1)
    offsets[1] = one(Ti)
    @inbounds for k in 1:coarse_nnz
        offsets[k + 1] = offsets[k] + counts[k]
    end
    ntriples = Int(offsets[end] - one(Ti))
    cursor = counts
    copyto!(cursor, 1, offsets, 1, coarse_nnz)
    pleft_map = host_buffer(old_left, Ti, ntriples)
    a_map = host_buffer(old_a, Ti, ntriples)
    pright_map = host_buffer(old_right, Ti, ntriples)
    with_setup_rows(nc) do II, tid
        I = Ti(II)
        @inbounds for dest in rowptr[I]:(rowptr[I + 1] - 1)
            markers[colval[dest], tid] = dest
        end
        @inbounds for q in Pt.offsets[I]:(Pt.offsets[I + 1] - 1)
            i = Pt.fine_rows[q]
            pleft = Pt.p_indices[q]
            for aidx in nzrange(A, i)
                j = A.colval[aidx]
                for pright in P.rowptr[j]:(P.rowptr[j + 1] - 1)
                    dest = markers[P.colval[pright], tid]
                    pos = cursor[dest]
                    pleft_map[pos] = pleft
                    a_map[pos] = aidx
                    pright_map[pos] = pright
                    cursor[dest] += one(Ti)
                end
            end
        end
    end
    map = GalerkinMap{Ti, typeof(offsets), typeof(pleft_map), typeof(a_map), typeof(pright_map)}(
        offsets, pleft_map, a_map, pright_map
    )
    return Ac, map, same_pattern
end

@inline function buffer_backend_matches(buffer, backend)
    return try
        same_backend(KernelAbstractions.get_backend(buffer), backend)
    catch
        false
    end
end

"""Logical prefix retaining the full backend allocation for later reuse."""
struct BackendBufferPrefix{T, V, A} <: AbstractVector{T}
    view::V
    allocation::A
end

function BackendBufferPrefix(view::V, allocation::A) where {V, A}
    return BackendBufferPrefix{eltype(view), V, A}(view, allocation)
end

backend_view(buffer::BackendBufferPrefix) = getfield(buffer, :view)
backend_allocation(buffer::BackendBufferPrefix) = getfield(buffer, :allocation)

Base.size(buffer::BackendBufferPrefix) = size(backend_view(buffer))
Base.axes(buffer::BackendBufferPrefix) = axes(backend_view(buffer))
Base.length(buffer::BackendBufferPrefix) = length(backend_view(buffer))
Base.IndexStyle(::Type{<:BackendBufferPrefix{T, V}}) where {T, V} =
    IndexStyle(V)
@inline Base.getindex(buffer::BackendBufferPrefix, index::Int) =
    getindex(backend_view(buffer), index)
@inline Base.setindex!(buffer::BackendBufferPrefix, value, index::Int) =
    setindex!(backend_view(buffer), value, index)
Base.fill!(buffer::BackendBufferPrefix, value) =
    fill!(backend_view(buffer), value)
Base.Array(buffer::BackendBufferPrefix) =
    Array(backend_view(buffer))
Base.copy(buffer::BackendBufferPrefix) =
    copy(backend_view(buffer))
function Base.copyto!(
        destination::BackendBufferPrefix,
        source::AbstractArray
    )
    copyto!(backend_view(destination), source)
    return destination
end
function Base.copyto!(
        destination::AbstractArray,
        source::BackendBufferPrefix
    )
    copyto!(destination, backend_view(source))
    return destination
end
function Base.copyto!(
        destination::BackendBufferPrefix,
        source::BackendBufferPrefix
    )
    copyto!(backend_view(destination), backend_view(source))
    return destination
end
function Base.copyto!(
        destination::BackendBufferPrefix,
        destination_offset::Integer, source::AbstractArray,
        source_offset::Integer, count::Integer
    )
    copyto!(
        backend_view(destination), destination_offset,
        source, source_offset, count
    )
    return destination
end
function Base.copyto!(
        destination::AbstractArray,
        destination_offset::Integer, source::BackendBufferPrefix,
        source_offset::Integer, count::Integer
    )
    copyto!(
        destination, destination_offset,
        backend_view(source), source_offset, count
    )
    return destination
end
function Base.copyto!(
        destination::BackendBufferPrefix,
        destination_offset::Integer, source::BackendBufferPrefix,
        source_offset::Integer, count::Integer
    )
    copyto!(
        backend_view(destination), destination_offset,
        backend_view(source), source_offset, count
    )
    return destination
end
KernelAbstractions.get_backend(buffer::BackendBufferPrefix) =
    KernelAbstractions.get_backend(backend_view(buffer))
logical_backend_buffer(buffer::BackendBufferPrefix) =
    backend_view(buffer)
function Adapt.adapt_structure(to, buffer::BackendBufferPrefix)
    return BackendBufferPrefix(
        Adapt.adapt(to, backend_view(buffer)),
        Adapt.adapt(to, backend_allocation(buffer))
    )
end

"""Return the allocation backing a contiguous prefix used for reuse."""
reusable_buffer(buffer::BackendBufferPrefix) =
    backend_allocation(buffer)

function reusable_buffer(buffer::SubArray{<:Any, 1})
    indices = parentindices(buffer)
    if length(indices) == 1 && only(indices) isa AbstractUnitRange &&
            first(only(indices)) == firstindex(parent(buffer))
        return parent(buffer)
    end
    return buffer
end

reusable_buffer(buffer) = buffer

backend_buffer_bytes(::Nothing) = 0
function backend_buffer_bytes(buffer::AbstractArray)
    allocation = reusable_buffer(buffer)
    return sizeof(eltype(allocation)) * length(allocation)
end
backend_buffer_bytes(buffer) = 0

"""Use the first `n` entries of a backend buffer without reallocating it."""
function buffer_prefix(buffer, n::Integer)
    n == length(buffer) && return buffer
    prefix = view(buffer, firstindex(buffer):(firstindex(buffer) + n - 1))
    return BackendBufferPrefix(prefix, buffer)
end

mark_backend_reallocation!(::Nothing, bytes::Integer) = nothing
function mark_backend_reallocation!(tracker, bytes::Integer)
    bytes > 0 || return nothing
    tracker[] += bytes
    return nothing
end

"""Capacity used when an existing backend buffer has to grow."""
function grown_backend_capacity(current::Integer, required::Integer)
    required <= current && return Int(current)
    # Leave modest headroom after a growth. AMG coarse-grid sizes tend to move
    # by a few percent between nonlinear updates, so exact-size allocations
    # otherwise replace the same buffers over and over.
    from_current = current + max(cld(current, 4), 1)
    from_required = required + max(cld(required, 8), 1)
    return Int(max(from_current, from_required))
end

function allocate_grown_backend_buffer(
        src::AbstractVector, backend,
        current_capacity::Integer, reallocation_tracker
    )
    capacity = grown_backend_capacity(current_capacity, length(src))
    allocation = backend_zeros(backend, eltype(src), capacity)
    destination = buffer_prefix(allocation, length(src))
    copyto!(destination, 1, host_copy_source(src), 1, length(src))
    replaced_bytes = current_capacity * sizeof(eltype(src))
    mark_backend_reallocation!(reallocation_tracker, replaced_bytes)
    return destination
end

function reusable_backend_allocation(old, ::Type{T}, backend) where {T}
    allocation = reusable_buffer(old)
    compatible = allocation isa AbstractVector &&
        eltype(allocation) === T &&
        buffer_backend_matches(allocation, backend)
    return allocation, compatible
end

function resize_backend_vector!(buffer::Vector, n::Integer)
    if JULIA_VER_CAN_SHRINK
        sizehint!(buffer, n; shrink = false)
    else
        sizehint!(buffer, n)
    end
    resize!(buffer, n)
    return buffer
end

"""Copy into an old backend allocation when its type and capacity permit."""
function copy_reusing(
        old, src::AbstractVector, backend;
        reallocation_tracker = nothing
    )
    old === src && return old
    allocation, compatible = reusable_backend_allocation(
        old, eltype(src), backend
    )
    if compatible && allocation isa Vector
        resize_backend_vector!(allocation, length(src))
        copyto!(allocation, host_copy_source(src))
        return allocation
    elseif compatible && length(allocation) >= length(src)
        destination = buffer_prefix(allocation, length(src))
        copyto!(destination, 1, host_copy_source(src), 1, length(src))
        return destination
    elseif compatible
        return allocate_grown_backend_buffer(
            src, backend, length(allocation), reallocation_tracker
        )
    end
    destination = backend_copy(backend, src)
    mark_backend_reallocation!(
        reallocation_tracker, backend_buffer_bytes(old)
    )
    return destination
end

function copy_optional_reusing(
        old, source, backend;
        reallocation_tracker = nothing
    )
    isnothing(source) && return nothing
    return copy_reusing(
        old, source, backend;
        reallocation_tracker = reallocation_tracker
    )
end

function zeros_reusing(
        old, backend, ::Type{T}, n::Integer;
        reallocation_tracker = nothing
    ) where {T}
    n = Int(n)
    allocation, compatible = reusable_backend_allocation(old, T, backend)
    if compatible && allocation isa Vector
        resize_backend_vector!(allocation, n)
        fill!(allocation, zero(T))
        return allocation
    elseif compatible && length(allocation) >= n
        destination = buffer_prefix(allocation, n)
        fill!(destination, zero(T))
        return destination
    elseif compatible
        capacity = grown_backend_capacity(length(allocation), n)
        destination = buffer_prefix(backend_zeros(backend, T, capacity), n)
        mark_backend_reallocation!(
            reallocation_tracker,
            length(allocation) * sizeof(T)
        )
        return destination
    end
    destination = backend_zeros(backend, T, n)
    mark_backend_reallocation!(
        reallocation_tracker, backend_buffer_bytes(old)
    )
    return destination
end

function owned_on_backend_reusing(
        A::StaticSparsityMatrixCSR, backend, block_size, old;
        reallocation_tracker = nothing
    )
    source = isnothing(old) ? host_csr(A) : A
    rp = copy_reusing(
        optional_property(old, :rowptr), source.rowptr, backend;
        reallocation_tracker = reallocation_tracker
    )
    cv = copy_reusing(
        optional_property(old, :colval), source.colval, backend;
        reallocation_tracker = reallocation_tracker
    )
    av = copy_reusing(
        optional_property(old, :nzval), source.nzval, backend;
        reallocation_tracker = reallocation_tracker
    )
    return csr_matrix(
        rp, cv, av, matrix_nrows(source), matrix_ncols(source);
        backend = backend, block_size = block_size
    )
end

function device_prolongation_reusing(
        P::Prolongation{Tv, Ti}, backend, old;
        reallocation_tracker = nothing
    ) where {Tv, Ti}
    rp = copy_reusing(
        optional_property(old, :rowptr), P.rowptr, backend;
        reallocation_tracker = reallocation_tracker
    )
    cv = copy_reusing(
        optional_property(old, :colval), P.colval, backend;
        reallocation_tracker = reallocation_tracker
    )
    pv = copy_reusing(
        optional_property(old, :nzval), P.nzval, backend;
        reallocation_tracker = reallocation_tracker
    )
    return Prolongation{Tv, Ti, typeof(rp), typeof(cv), typeof(pv)}(rp, cv, pv, P.nrow, P.ncol)
end

function device_transpose_reusing(
        M::TransposeMap{Ti}, backend, old;
        reallocation_tracker = nothing
    ) where {Ti}
    o = copy_reusing(
        optional_property(old, :offsets), M.offsets, backend;
        reallocation_tracker = reallocation_tracker
    )
    r = copy_reusing(
        optional_property(old, :fine_rows), M.fine_rows, backend;
        reallocation_tracker = reallocation_tracker
    )
    p = copy_reusing(
        optional_property(old, :p_indices), M.p_indices, backend;
        reallocation_tracker = reallocation_tracker
    )
    return TransposeMap{Ti, typeof(o), typeof(r), typeof(p)}(o, r, p)
end

function device_galerkin_reusing(
        M::GalerkinMap{Ti}, backend, old;
        reallocation_tracker = nothing
    ) where {Ti}
    o = copy_reusing(
        optional_property(old, :offsets), M.offsets, backend;
        reallocation_tracker = reallocation_tracker
    )
    l = copy_reusing(
        optional_property(old, :p_left), M.p_left, backend;
        reallocation_tracker = reallocation_tracker
    )
    a = copy_reusing(
        optional_property(old, :a_index), M.a_index, backend;
        reallocation_tracker = reallocation_tracker
    )
    r = copy_reusing(
        optional_property(old, :p_right), M.p_right, backend;
        reallocation_tracker = reallocation_tracker
    )
    return GalerkinMap{Ti, typeof(o), typeof(l), typeof(a), typeof(r)}(o, l, a, r)
end

native_dense_lu(::Any) = false
native_dense_lu(buffer::BackendBufferPrefix) =
    native_dense_lu(reusable_buffer(buffer))

function build_host_coarse_solver(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    C = host_csr(A)
    matrix = zeros(Tv, matrix_nrows(C), matrix_ncols(C))
    @inbounds for i in 1:matrix_nrows(C), k in C.rowptr[i]:(C.rowptr[i + 1] - 1)
        matrix[i, C.colval[k]] = C.nzval[k]
    end
    return HostLUState(lu!(matrix), C.nzval, C.rowptr, C.colval, zeros(Tv, matrix_nrows(C)))
end

function build_coarse_solver(A::StaticSparsityMatrixCSR{Tv, Ti}, backend, reuse = nothing) where {Tv, Ti}
    if backend isa KernelAbstractions.CPU && Tv <: Union{Float64, ComplexF64}
        matrix = sparse_matrix(A)
        csr_to_csc = Vector{Ti}(undef, matrix_nonzeros(A))
        @inbounds for j in 1:matrix_ncols(A)
            col = Ti(j)
            for q in nzrange(matrix, j)
                row = matrix.rowval[q]
                csr_to_csc[find_column(A, row, col)] = Ti(q)
            end
        end
        return CoarseLUState(matrix, lu(matrix), csr_to_csc)
    end
    if !(backend isa KernelAbstractions.CPU) && Tv <: Number
        if reuse isa SparseLU && sparse_lu_same_pattern(reuse, A)
            return resetup_sparse_lu!(reuse, A)
        end
        return setup_sparse_lu(A)
    end
    if backend isa KernelAbstractions.CPU || native_dense_lu(A.nzval)
        if reuse isa DenseLUState && size(reuse.factorization.factors) == size(A)
            return update_coarse_solver!(reuse, A)
        end
        matrix = KernelAbstractions.zeros(backend, Tv, matrix_nrows(A), matrix_ncols(A))
        copy_csr_to_dense!(matrix, A)
        return DenseLUState(lu!(matrix))
    end
    return build_host_coarse_solver(A)
end

function make_level(
        A_cpu::StaticSparsityMatrixCSR{Tv, Ti}, P_cpu, Pt_cpu, G_cpu, cf, cmap,
        strong, backend, options, old_level = nothing;
        reuse_A_structure::Bool = false,
        reallocation_tracker = nothing
    ) where {Tv, Ti}
    if isnothing(old_level) && backend isa KernelAbstractions.CPU
        # The symbolic setup has already created owned CPU arrays. Retain them
        # directly instead of duplicating the complete hierarchy.
        A, P, Pt, G = A_cpu, P_cpu, Pt_cpu, G_cpu
    else
        old_A = optional_property(old_level, :A)
        old_P = optional_property(old_level, :P)
        old_Pt = optional_property(old_level, :Pt)
        old_G = optional_property(old_level, :galerkin)
        A = if reuse_A_structure
            old_A
        else
            owned_on_backend_reusing(
                A_cpu, backend, options.block_size, old_A;
                reallocation_tracker = reallocation_tracker
            )
        end
        P = isnothing(P_cpu) ? nothing : device_prolongation_reusing(
                P_cpu, backend, old_P;
                reallocation_tracker = reallocation_tracker
            )
        Pt = isnothing(Pt_cpu) ? nothing : device_transpose_reusing(
                Pt_cpu, backend, old_Pt;
                reallocation_tracker = reallocation_tracker
            )
        G = isnothing(G_cpu) ? nothing : device_galerkin_reusing(
                G_cpu, backend, old_G;
                reallocation_tracker = reallocation_tracker
            )
    end
    old_smoother = optional_property(old_level, :smoother)
    S = setup_smoother(
        A, options.smoother; reuse = old_smoother,
        reallocation_tracker = reallocation_tracker
    )
    old_coarse_solver = optional_property(old_level, :coarse_solver)
    coarse_solver = if !isnothing(P_cpu) || options.coarse_solver != :lu
        nothing
    else
        build_coarse_solver(A, backend, old_coarse_solver)
    end
    old_r = optional_property(old_level, :residual)
    old_xc = optional_property(old_level, :correction)
    old_bc = optional_property(old_level, :rhs)
    r = zeros_reusing(
        old_r, backend, Tv, matrix_nrows(A);
        reallocation_tracker = reallocation_tracker
    )
    ncoarse = isnothing(P) ? matrix_nrows(A) : P.ncol
    xc = zeros_reusing(
        old_xc, backend, Tv, ncoarse;
        reallocation_tracker = reallocation_tracker
    )
    bc = zeros_reusing(
        old_bc, backend, Tv, ncoarse;
        reallocation_tracker = reallocation_tracker
    )
    if isnothing(old_level) && backend isa KernelAbstractions.CPU
        cf_d, cm_d, st_d = cf, cmap, strong
    else
        old_cf = optional_property(old_level, :cf)
        old_cm = optional_property(old_level, :coarse_map)
        old_st = optional_property(old_level, :strength)
        cf_d = copy_optional_reusing(
            old_cf, cf, backend;
            reallocation_tracker = reallocation_tracker
        )
        cm_d = copy_optional_reusing(
            old_cm, cmap, backend;
            reallocation_tracker = reallocation_tracker
        )
        st_d = copy_optional_reusing(
            old_st, strong, backend;
            reallocation_tracker = reallocation_tracker
        )
    end
    return AMGLevel{Tv, Ti}(A, P, Pt, G, S, coarse_solver, r, xc, bc, cf_d, cm_d, st_d)
end

function reuse_level(reuse_levels, level_index)
    return if isnothing(reuse_levels) || level_index > length(reuse_levels)
        nothing
    else
        reuse_levels[level_index]
    end
end

function setup_reuse_buffer(
        cpu_backend, old_level, field::Symbol,
        staged_buffer
    )
    return cpu_backend ? optional_property(old_level, field) : staged_buffer
end

next_stage_matrix(workspace, stage_slot) =
    stage_slot == 1 ? workspace.stage_matrix2 : workspace.stage_matrix1

function store_stage_matrix!(workspace, stage_slot, matrix)
    if stage_slot == 1
        workspace.stage_matrix2 = matrix
        return 2
    else
        workspace.stage_matrix1 = matrix
        return 1
    end
end

function validate_setup_options(options::AMGOptions)
    options.cycle in (:V, :W) || throw(ArgumentError("cycle must be :V or :W"))
    options.max_levels >= 1 || throw(ArgumentError("max_levels must be positive"))
    options.coarse_size >= 1 || throw(ArgumentError("coarse_size must be positive"))
    options.max_row_sum >= 0 || throw(ArgumentError("max_row_sum must be non-negative"))
    options.coarse_solver in (:lu, :spai0) ||
        throw(ArgumentError("coarse_solver must be :lu or :spai0"))
    aggregation = options.coarsening isa Aggregation
    constant = options.interpolation isa ConstantInterpolation
    aggregation == constant || throw(
        ArgumentError(
            "ConstantInterpolation must be used with Aggregation, and Aggregation requires it"
        )
    )
    return nothing
end

function initial_hierarchy_state(
        Ain::StaticSparsityMatrixCSR, reuse_levels,
        workspace, host_finest
    )
    backend = matrix_backend(Ain)
    cpu_backend = backend isa KernelAbstractions.CPU
    if !isnothing(host_finest)
        current = host_finest
        stage_slot = cpu_backend ? 0 : 1
    elseif cpu_backend
        old_first = optional_property(reuse_level(reuse_levels, 1), :A)
        current = host_csr_reusing(Ain, old_first)
        stage_slot = 0
    else
        current = host_csr_reusing(Ain, workspace.stage_matrix1)
        workspace.stage_matrix1 = current
        stage_slot = 1
    end
    return current, stage_slot, backend, cpu_backend
end

function build_hierarchy(
        Ain::StaticSparsityMatrixCSR{Tv, Ti}, options::AMGOptions;
        reuse_levels = nothing,
        workspace = SetupWorkspace(Tv, Ti),
        host_finest = nothing,
        reuse_finest_structure::Bool = false,
        reallocation_tracker = nothing
    ) where {Tv, Ti}
    validate_setup_options(options)
    current, stage_slot, backend, cpu_backend =
        initial_hierarchy_state(Ain, reuse_levels, workspace, host_finest)
    levels = AMGLevel{Tv, Ti}[]
    pattern_matches_old = reuse_finest_structure
    for level_index in 1:(options.max_levels - 1)
        n = matrix_nrows(current)
        n <= options.coarse_size && break
        number_of_nonzeros = matrix_nonzeros(current)
        old = reuse_level(reuse_levels, level_index)
        symbolic_old = cpu_backend ? old : nothing
        reusable_split = pattern_matches_old && !isnothing(old) &&
            !isnothing(old.P) && size(old.A) == size(current) &&
            matrix_nonzeros(old.A) == number_of_nonzeros
        old_strength = if reusable_split && cpu_backend
            workspace.bool1
        else
            setup_reuse_buffer(
                cpu_backend, symbolic_old, :strength,
                workspace.stage_strength
            )
        end
        old_cf = setup_reuse_buffer(
            cpu_backend, symbolic_old, :cf, workspace.stage_cf
        )
        old_map = setup_reuse_buffer(
            cpu_backend, symbolic_old, :coarse_map,
            workspace.stage_coarse_map
        )
        theta = options.coarsening.theta
        strong = strength(current, theta, options.max_row_sum, old_strength)
        cpu_backend || (workspace.stage_strength = strong)
        reuse_split = false
        if reusable_split
            old_strong = old.strength
            if cpu_backend
                reuse_split = same_prefix(
                    strong, old_strong,
                    number_of_nonzeros
                )
            else
                old_host = host_buffer(
                    workspace.bool1, Bool, number_of_nonzeros
                )
                copyto!(old_host, 1, old_strong, 1, number_of_nonzeros)
                reuse_split = same_prefix(
                    strong, old_host,
                    number_of_nonzeros
                )
            end
        end
        if reuse_split
            if cpu_backend
                cf, cmap = old.cf, old.coarse_map
            else
                cf = host_prefix_reusing(workspace.stage_cf, old.cf, n)
                cmap = host_prefix_reusing(
                    workspace.stage_coarse_map,
                    old.coarse_map, n
                )
                workspace.stage_cf = cf
                workspace.stage_coarse_map = cmap
            end
            nc = old.P.ncol
        elseif options.coarsening isa Aggregation
            cf, cmap, nc = aggregation_split(
                current, strong, old_cf, old_map,
                workspace
            )
        else
            cf, cmap, nc = cf_split(
                current, strong, options.coarsening,
                old_cf, old_map, workspace
            )
            nc == 0 && break
        end
        old_P = setup_reuse_buffer(
            cpu_backend, symbolic_old, :P, workspace.stage_prolongation
        )
        P = build_prolongation(
            current, cf, cmap, nc, strong,
            options.interpolation, old_P, workspace
        )
        if !cpu_backend
            workspace.stage_cf = cf
            workspace.stage_coarse_map = cmap
            workspace.stage_prolongation = P
        end
        nc == 0 && break
        nc >= n && break
        old_Pt = setup_reuse_buffer(
            cpu_backend, symbolic_old, :Pt, workspace.stage_transpose
        )
        old_G = setup_reuse_buffer(
            cpu_backend, symbolic_old, :galerkin, workspace.stage_galerkin
        )
        if cpu_backend
            old_coarse_level = reuse_level(reuse_levels, level_index + 1)
            old_coarse = optional_property(old_coarse_level, :A)
        else
            old_coarse = next_stage_matrix(workspace, stage_slot)
        end
        Pt = transpose_map(P, old_Pt, workspace)
        coarse, G, staging_pattern_match =
            galerkin_structure(current, P, Pt, old_G, old_coarse, workspace)
        if cpu_backend
            pattern_matches_old = staging_pattern_match
        else
            old_next = reuse_level(reuse_levels, level_index + 1)
            pattern_matches_old = !isnothing(old_next) &&
                same_csr_pattern(coarse, old_next.A, workspace)
        end
        push!(
            levels, make_level(
                current, P, Pt, G, cf, cmap, strong, backend,
                options, old;
                reuse_A_structure = reuse_finest_structure && level_index == 1,
                reallocation_tracker = reallocation_tracker
            )
        )
        if !cpu_backend
            workspace.stage_transpose = Pt
            workspace.stage_galerkin = G
            stage_slot = store_stage_matrix!(workspace, stage_slot, coarse)
        end
        current = coarse
    end
    level_index = length(levels) + 1
    old = reuse_level(reuse_levels, level_index)
    push!(
        levels, make_level(
            current, nothing, nothing, nothing, nothing, nothing,
            nothing, backend, options, old;
            reallocation_tracker = reallocation_tracker
        )
    )
    synchronize_backend(backend)
    return levels
end

"""
    setup_amg(A, options=AMGOptions())

Build an owned, reusable AMG hierarchy. Numerical kernels and all hierarchy
arrays use the backend of `A` (or the backend requested when calling
`csr_matrix`).
"""
function setup_amg(A::StaticSparsityMatrixCSR{Tv, Ti}, options::AMGOptions = AMGOptions()) where {Tv, Ti}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("AMG requires a square matrix"))
    workspace = SetupWorkspace(Tv, Ti)
    levels = build_hierarchy(A, options; workspace = workspace)
    return AMGHierarchy{Tv, Ti}(
        levels, workspace, options, matrix_backend(A),
        matrix_batch_size(levels[1].A),
        host_prefix(A.rowptr, matrix_nrows(A) + 1),
        host_prefix(A.colval, matrix_nonzeros(A)), 0, Inf, 0
    )
end

setup_amg(A::SparseMatrixCSC, options::AMGOptions = AMGOptions()) =
    setup_amg(csr_matrix(A; block_size = options.block_size), options)

setup_amg(A, coarsening::AbstractCoarsening; kwargs...) =
    setup_amg(A, AMGOptions(; coarsening = coarsening, kwargs...))
