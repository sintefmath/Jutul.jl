# Symbolic setup on the matrix backend. Scans and reductions use the backend
# array implementation; only scalar sizes and termination flags reach the host.
function pmis_vectors_match(first, second)
    return first === second || mapreduce(
        ==, &, logical_backend_buffer(first), logical_backend_buffer(second); init = true
    )
end

function pmis_buffer(old, backend, ::Type{T}, n, tracker) where {T}
    allocation, compatible = reusable_backend_allocation(old, T, backend)
    if compatible && allocation isa Vector
        return resize_backend_vector!(allocation, Int(n))
    elseif compatible && length(allocation) >= n
        return length(old) == n ? old : buffer_prefix(allocation, n)
    end
    if backend isa KernelAbstractions.CPU
        return Vector{T}(undef, n)
    end
    capacity = isnothing(old) ? Int(n) :
        grown_backend_capacity(compatible ? length(allocation) : 0, n)
    mark_backend_reallocation!(tracker, backend_buffer_bytes(old))
    return buffer_prefix(KernelAbstractions.allocate(backend, T, capacity), n)
end

function pmis_scratch!(workspace, key, backend, ::Type{T}, n, tracker) where {T}
    old = get(workspace.pmis_buffers, key, nothing)
    buffer = pmis_buffer(old, backend, T, n, tracker)
    workspace.pmis_buffers[key] = reusable_buffer(buffer)
    return logical_backend_buffer(buffer)
end

function pmis_scan!(offsets, counts, backend)
    cumsum!(logical_backend_buffer(offsets), counts)
    last = Vector{eltype(counts)}(undef, 1)
    synchronize_backend(backend)
    copyto!(last, 1, logical_backend_buffer(offsets), length(counts), 1)
    return Int(last[1] - 1)
end

@inline pmis_active(::Nothing, k) = true
@inline pmis_active(mask, k) = @inbounds mask[k]

@kernel function pmis_transpose_counts!(counts, @Const(rp), @Const(cv), @Const(mask))
    i = @index(Global)
    @inbounds for k in rp[i]:(rp[i + 1] - 1)
        if pmis_active(mask, k)
            j = cv[k] + 1
            KernelAbstractions.@atomic counts[j] += 1
        end
    end
end

@inline pmis_store_index!(::Nothing, slot, k) = nothing
@inline pmis_store_index!(indices, slot, k) = (@inbounds indices[slot] = k)

@kernel function pmis_transpose_fill!(rows, indices, counts, @Const(offsets), @Const(rp), @Const(cv), @Const(mask))
    i = @index(Global)
    @inbounds for k in rp[i]:(rp[i + 1] - 1)
        if pmis_active(mask, k)
            j = cv[k] + 1
            next = KernelAbstractions.@atomic counts[j] += 1
            slot = offsets[j - 1] + next - 1
            rows[slot] = i
            pmis_store_index!(indices, slot, k)
        end
    end
end

function pmis_transpose(
        rp, cv, mask, nrow, ncol, backend, block, workspace, old, tracker
    )
    Ti = eltype(cv)
    counts = pmis_scratch!(workspace, :transpose_counts, backend, Int, ncol + 1, tracker)
    fill!(counts, 0)
    # Shift the row counts by one and start the inclusive scan at one for CSR.
    fill_kernel!(backend, block)(counts, 1, 1; ndrange = 1)
    pmis_transpose_counts!(backend, block)(counts, rp, cv, mask; ndrange = nrow)
    is_strength = !isnothing(mask)
    offsets = if is_strength
        pmis_scratch!(workspace, :strength_offsets, backend, Int, ncol + 1, tracker)
    else
        pmis_buffer(optional_property(old, :offsets), backend, Int, ncol + 1, tracker)
    end
    nentries = if is_strength
        pmis_scan!(offsets, counts, backend)
    else
        cumsum!(logical_backend_buffer(offsets), counts)
        length(cv)
    end
    rows = if is_strength
        pmis_scratch!(workspace, :strength_rows, backend, Ti, nentries, tracker)
    else
        pmis_buffer(optional_property(old, :fine_rows), backend, Ti, nentries, tracker)
    end
    indices = is_strength ? nothing :
        pmis_buffer(optional_property(old, :p_indices), backend, Ti, nentries, tracker)
    fill!(counts, 0)
    pmis_transpose_fill!(backend, block)(rows, indices, counts, offsets, rp, cv, mask; ndrange = nrow)
    return offsets, rows, indices
end

@inline function pmis_hash(i)
    x = i % UInt32
    x = (x ⊻ (x >> 16)) * 0x7feb352d
    x = (x ⊻ (x >> 15)) * 0x846ca68b
    return x ⊻ (x >> 16)
end

@inline function pmis_higher_priority(i, j, offsets)
    @inbounds di, dj = offsets[i + 1] - offsets[i], offsets[j + 1] - offsets[j]
    # In-degree prioritizes points depended on by many neighbors. The fixed
    # hash breaks ties without depending on traversal or atomic insertion order.
    hi, hj = pmis_hash(i), pmis_hash(j)
    return di > dj || (di == dj && (hi > hj || (hi == hj && i > j)))
end

@kernel function pmis_initialize!(cf, @Const(rp), @Const(strong), @Const(offsets))
    i = @index(Global)
    @inbounds begin
        has_out = false
        for k in rp[i]:(rp[i + 1] - 1)
            has_out |= strong[k]
        end
        # HYPRE's special F points have no outgoing strong dependencies,
        # even when other rows depend on them. Points influencing no other
        # row are ordinary F points and must not enter the independent set.
        cf[i] = !has_out ? Int8(-3) :
            (offsets[i] == offsets[i + 1] ? Int8(-1) : Int8(0))
    end
end

@kernel function pmis_select!(selected, @Const(cf), @Const(rp), @Const(cv), @Const(strong), @Const(offsets), @Const(rows))
    i = @index(Global)
    @inbounds begin
        wins = iszero(cf[i])
        if wins
            for k in rp[i]:(rp[i + 1] - 1)
                j = cv[k]
                if strong[k] && iszero(cf[j]) && pmis_higher_priority(j, i, offsets)
                    wins = false
                    break
                end
            end
            if wins
                for k in offsets[i]:(offsets[i + 1] - 1)
                    j = rows[k]
                    if iszero(cf[j]) && pmis_higher_priority(j, i, offsets)
                        wins = false
                        break
                    end
                end
            end
        end
        selected[i] = wins
    end
end

@kernel function pmis_commit!(cf, pending, @Const(selected), @Const(rp), @Const(cv), @Const(strong))
    i = @index(Global)
    @inbounds begin
        state = cf[i]
        if selected[i]
            state = Int8(1)
        elseif iszero(state)
            covered = false
            for k in rp[i]:(rp[i + 1] - 1)
                covered |= strong[k] && selected[cv[k]]
            end
            # Coverage follows outgoing dependencies, as interpolation does.
            # An incoming edge alone does not make this row an F point.
            covered && (state = Int8(-1))
        end
        cf[i] = state
        pending[i] = iszero(state)
    end
end

@kernel function pmis_coarse_counts!(counts, @Const(cf))
    i = @index(Global)
    @inbounds begin
        counts[i + 1] = Int(cf[i] == 1)
        i == 1 && (counts[i] = 1)
    end
end

@kernel function pmis_coarse_map!(map, @Const(offsets), @Const(cf))
    i = @index(Global)
    @inbounds map[i] = cf[i] == 1 ? offsets[i] : 0
end

function pmis_split(A, strong, interpolation, workspace, old, tracker)
    backend, block = matrix_backend(A), matrix_kernel_block_size(A)
    n = matrix_nrows(A)
    offsets, rows, _ = pmis_transpose(
        A.rowptr, A.colval, strong, n, n, backend, block, workspace, nothing, tracker
    )
    cf = pmis_buffer(optional_property(old, :cf), backend, Int8, n, tracker)
    selected = pmis_scratch!(workspace, :selected, backend, Bool, n, tracker)
    pending = pmis_scratch!(workspace, :pending, backend, Bool, n, tracker)
    pmis_initialize!(backend, block)(cf, A.rowptr, strong, offsets; ndrange = n)
    select! = pmis_select!(backend, block)
    commit! = pmis_commit!(backend, block)
    while true
        select!(selected, cf, A.rowptr, A.colval, strong, offsets, rows; ndrange = n)
        commit!(cf, pending, selected, A.rowptr, A.colval, strong; ndrange = n)
        any(pending) || break
    end
    counts = pmis_scratch!(workspace, :row_counts, backend, Int, n + 1, tracker)
    scan = pmis_scratch!(workspace, :row_scan, backend, Int, n + 1, tracker)
    pmis_coarse_counts!(backend, block)(counts, cf; ndrange = n)
    nc = pmis_scan!(scan, counts, backend)
    cmap = pmis_buffer(optional_property(old, :coarse_map), backend, eltype(A.colval), n, tracker)
    pmis_coarse_map!(backend, block)(cmap, scan, cf; ndrange = n)
    return cf, cmap, nc
end

@kernel function pmis_p_candidates!(counts, @Const(rp), @Const(cv), @Const(cf), @Const(strong), extended)
    i = @index(Global)
    @inbounds begin
        count = Int(cf[i] == 1)
        if cf[i] == -1
            for k in rp[i]:(rp[i + 1] - 1)
                strong[k] || continue
                j = cv[k]
                if cf[j] == 1
                    count += 1
                elseif extended && cf[j] == -1
                    for q in rp[j]:(rp[j + 1] - 1)
                        count += Int(strong[q] && cf[cv[q]] == 1)
                    end
                end
            end
        end
        counts[i + 1] = count
        i == 1 && (counts[i] = 1)
    end
end

@inline function pmis_candidate_position(cols, first, last, c)
    @inbounds for k in first:last
        cols[k] == c && return k
    end
    return 0
end

@inline function pmis_insert_candidate!(cols, first, last, c)
    iszero(pmis_candidate_position(cols, first, last, c)) || return last
    @inbounds cols[last + 1] = c
    return last + 1
end

@inline function pmis_candidate_weights!(
        vals, cols, first, last, rp, cv, av, cf, cmap, strong, i, extended
    )
    T = eltype(vals)
    diagonal = zero(T)
    @inbounds for p in first:last
        vals[p] = zero(T)
    end
    # The complete candidate set is already in cols. Distribute each strong
    # fine neighbor's contribution once across that set.
    @inbounds for k in rp[i]:(rp[i + 1] - 1)
        j, aij = cv[k], av[k]
        pos = cf[j] == 1 ? pmis_candidate_position(cols, first, last, cmap[j]) : 0
        if j == i
            diagonal += aij
        elseif !iszero(pos)
            vals[pos] += aij
        elseif strong[k] && cf[j] == -1
            diag_j = zero(T)
            for q in rp[j]:(rp[j + 1] - 1)
                cv[q] == j && (diag_j += av[q])
            end
            sign_j = diag_j < 0 ? -one(T) : one(T)
            denominator = zero(T)
            for q in rp[j]:(rp[j + 1] - 1)
                l, ajl = cv[q], av[q]
                if l != j && sign_j * ajl < 0 &&
                        (
                        (extended && l == i) ||
                            (cf[l] == 1 && !iszero(pmis_candidate_position(cols, first, last, cmap[l])))
                    )
                    denominator += ajl
                end
            end
            if iszero(denominator)
                diagonal += aij
            else
                distribute = aij / denominator
                for q in rp[j]:(rp[j + 1] - 1)
                    l, ajl = cv[q], av[q]
                    (l != j && sign_j * ajl < 0) || continue
                    if extended && l == i
                        diagonal += distribute * ajl
                    elseif cf[l] == 1
                        pos = pmis_candidate_position(cols, first, last, cmap[l])
                        iszero(pos) || (vals[pos] += distribute * ajl)
                    end
                end
            end
        elseif cf[j] != -3
            diagonal += aij
        end
    end
    scale = iszero(diagonal) ? (extended ? one(T) : zero(T)) : -inv(diagonal)
    total = zero(T)
    @inbounds for p in first:last
        vals[p] *= scale
        total += vals[p]
    end
    return total
end

@kernel function pmis_p_select!(
        cols, vals, counts, @Const(slots), @Const(rp), @Const(cv), @Const(av),
        @Const(cf), @Const(cmap), @Const(strong), extended, truncation, max_elements, norm_p, rescale
    )
    i = @index(Global)
    @inbounds begin
        first = slots[i]
        last = first - 1
        if cf[i] == 1
            last = first
            cols[first] = cmap[i]
            vals[first] = one(eltype(vals))
        elseif cf[i] == -1
            for k in rp[i]:(rp[i + 1] - 1)
                strong[k] || continue
                j = cv[k]
                if cf[j] == 1
                    last = pmis_insert_candidate!(cols, first, last, cmap[j])
                elseif extended && cf[j] == -1
                    for q in rp[j]:(rp[j + 1] - 1)
                        l = cv[q]
                        if strong[q] && cf[l] == 1
                            last = pmis_insert_candidate!(cols, first, last, cmap[l])
                        end
                    end
                end
            end
            total = pmis_candidate_weights!(
                vals, cols, first, last, rp, cv, av, cf, cmap, strong, i, extended
            )
            # Rank by weight, then pack the retained entries in CSR column order.
            for k in (first + 1):last
                col, val = cols[k], vals[k]
                j = k
                while j > first && (
                        abs(val) > abs(vals[j - 1]) ||
                            (abs(val) == abs(vals[j - 1]) && col < cols[j - 1])
                    )
                    cols[j], vals[j] = cols[j - 1], vals[j - 1]
                    j -= 1
                end
                cols[j], vals[j] = col, val
            end
            max_elements > 0 && (last = min(last, first + max_elements - 1))
            kept = zero(total)
            leading = first <= last ? abs(vals[first]) : zero(total)
            for k in first:last
                if truncation > 0 && !iszero(leading) &&
                        candidate_score(vals[k] / leading, norm_p) < truncation
                    last = k - 1
                    break
                end
                kept += vals[k]
            end
            scale = rescale && !iszero(kept) ? total / kept : one(total)
            for k in first:last
                col, val = cols[k], scale * vals[k]
                j = k
                while j > first && col < cols[j - 1]
                    cols[j], vals[j] = cols[j - 1], vals[j - 1]
                    j -= 1
                end
                cols[j], vals[j] = col, val
            end
        end
        counts[i + 1] = last - first + 1
        i == 1 && (counts[i] = 1)
    end
end

@kernel function pmis_p_pack!(cols, vals, @Const(rp), @Const(slots), @Const(candidate_cols), @Const(candidate_vals))
    i = @index(Global)
    @inbounds for k in rp[i]:(rp[i + 1] - 1)
        j = slots[i] + (k - rp[i])
        cols[k] = candidate_cols[j]
        vals[k] = candidate_vals[j]
    end
end

@kernel function pmis_p_pattern_matches!(
        matches, @Const(scan), @Const(slots), @Const(candidate_cols), @Const(rp), @Const(cv)
    )
    i = @index(Global)
    @inbounds begin
        same = scan[i] == rp[i] && scan[i + 1] == rp[i + 1]
        if same
            for k in rp[i]:(rp[i + 1] - 1)
                if cv[k] != candidate_cols[slots[i] + k - rp[i]]
                    same = false
                    break
                end
            end
        end
        matches[i] = same
    end
end

function pmis_index_limit(::Type{Ti}, n) where {Ti}
    n < typemax(Ti) || throw(ArgumentError("AMG sparsity exceeds the CSR index range; use wider indices"))
    return n
end

function pmis_offsets(::Type{Ti}, scan, backend, block, old, tracker) where {Ti}
    offsets = pmis_buffer(old, backend, Ti, length(scan), tracker)
    copy_kernel!(backend, block)(offsets, scan, length(scan); ndrange = length(scan))
    return offsets
end

function pmis_prolongation(
        A, cf, cmap, nc, strong, interpolation, workspace, old, compare_pattern, tracker
    )
    backend, block = matrix_backend(A), matrix_kernel_block_size(A)
    n = matrix_nrows(A)
    Tv, Ti = eltype(A), eltype(A.colval)
    counts = pmis_scratch!(workspace, :row_counts, backend, Int, n + 1, tracker)
    slots = pmis_scratch!(workspace, :candidate_slots, backend, Int, n + 1, tracker)
    extended = interpolation isa ExtendedIInterpolation
    pmis_p_candidates!(backend, block)(
        counts, A.rowptr, A.colval, cf, strong, extended; ndrange = n
    )
    capacity = pmis_scan!(slots, counts, backend)
    cols = pmis_scratch!(workspace, :candidate_cols, backend, Ti, capacity, tracker)
    vals = pmis_scratch!(workspace, :candidate_vals, backend, Tv, capacity, tracker)
    pmis_p_select!(backend, block)(
        cols, vals, counts, slots, A.rowptr, A.colval, A.nzval, cf, cmap, strong, extended,
        Tv(interpolation.truncation), interpolation.max_elements, interpolation.norm_p,
        interpolation.rescale; ndrange = n
    )
    scan = pmis_scratch!(workspace, :row_scan, backend, Int, n + 1, tracker)
    np = pmis_index_limit(Ti, pmis_scan!(scan, counts, backend))
    if compare_pattern && !isnothing(old) && old.ncol == nc && length(old.colval) == np
        matches = pmis_scratch!(workspace, :matching_p_rows, backend, Bool, n, tracker)
        pmis_p_pattern_matches!(backend, block)(
            matches, scan, slots, cols, old.rowptr, old.colval; ndrange = n
        )
        if all(matches)
            # Candidate selection can change with the values even when the
            # strength graph stays fixed. Reuse the maps only after this check.
            pmis_p_pack!(backend, block)(
                old.colval, old.nzval, old.rowptr, slots, cols, vals; ndrange = n
            )
            return old, true
        end
    end
    rp = pmis_offsets(Ti, scan, backend, block, optional_property(old, :rowptr), tracker)
    cv = pmis_buffer(optional_property(old, :colval), backend, Ti, np, tracker)
    pv = pmis_buffer(optional_property(old, :nzval), backend, Tv, np, tracker)
    pmis_p_pack!(backend, block)(cv, pv, rp, slots, cols, vals; ndrange = n)
    return Prolongation{Tv, Ti, typeof(rp), typeof(cv), typeof(pv)}(rp, cv, pv, n, nc), false
end

@kernel function pmis_galerkin_counts!(counts, @Const(trp), @Const(rows), @Const(arp), @Const(acv), @Const(prp))
    i = @index(Global)
    @inbounds begin
        count = 0
        for t in trp[i]:(trp[i + 1] - 1)
            r = rows[t]
            for a in arp[r]:(arp[r + 1] - 1)
                j = acv[a]
                count += prp[j + 1] - prp[j]
            end
        end
        counts[i + 1] = count
        i == 1 && (counts[i] = 1)
    end
end

@inline pmis_galerkin_less(a, b) = isless(a, b)

@inline function pmis_galerkin_less(a::NTuple{4, Int32}, b::NTuple{4, Int32})
    # CSR indices are positive. Each word compares two indices in lexicographic order.
    first_a = (UInt64(unsigned(a[1])) << 32) | unsigned(a[2])
    first_b = (UInt64(unsigned(b[1])) << 32) | unsigned(b[2])
    first_a == first_b || return first_a < first_b
    second_a = (UInt64(unsigned(a[3])) << 32) | unsigned(a[4])
    second_b = (UInt64(unsigned(b[3])) << 32) | unsigned(b[4])
    return second_a < second_b
end

@inline function pmis_sift_down!(entries, base, root, last, value)
    @inbounds begin
        while root <= last ÷ 2
            child = 2 * root
            best = entries[base + child]
            if child < last
                right = entries[base + child + 1]
                if pmis_galerkin_less(best, right)
                    best = right
                    child += 1
                end
            end
            pmis_galerkin_less(value, best) || break
            entries[base + root] = best
            root = child
        end
        entries[base + root] = value
    end
    return nothing
end

@inline function pmis_heapsort_row!(entries, first, last)
    base, n = first - 1, last - first + 1
    @inbounds begin
        for r in (n ÷ 2):-1:1
            pmis_sift_down!(entries, base, r, n, entries[base + r])
        end
        for j in last:-1:(first + 1)
            root, value = entries[first], entries[j]
            entries[j] = root
            pmis_sift_down!(entries, base, 1, j - first, value)
        end
    end
    return nothing
end

@inline function pmis_fill_galerkin_row!(entries, first, trp, rows, pl, arp, acv, prp, pcv, i)
    slot = first
    Ti = eltype(pcv)
    @inbounds for t in trp[i]:(trp[i + 1] - 1)
        r = rows[t]
        for a in arp[r]:(arp[r + 1] - 1)
            j = acv[a]
            for p in prp[j]:(prp[j + 1] - 1)
                entries[slot] = (pcv[p], pl[t], Ti(a), Ti(p))
                slot += 1
            end
        end
    end
    return slot - 1
end

@inline function pmis_galerkin_column_count(entries, first, last, previous)
    count = 0
    @inbounds for t in first:last
        col = entries[t][1]
        count += Int(col != previous)
        previous = col
    end
    return count
end

function pmis_galerkin_rows_cpu!(
        entries, counts, slots, trp, rows, pl, arp, acv, prp, pcv, ncoarse, batch_size
    )
    counts[1] = 1
    foreach_cpu_row(ncoarse, batch_size) do i
        @inbounds begin
            first = slots[i]
            last = pmis_fill_galerkin_row!(entries, first, trp, rows, pl, arp, acv, prp, pcv, i)
            if last - first + 1 <= 1024
                sort!(view(entries, first:last); alg = QuickSort, lt = pmis_galerkin_less)
            else
                # Bound quicksort's worst-case cost on rows with high fill.
                pmis_heapsort_row!(entries, first, last)
            end
            counts[i + 1] = pmis_galerkin_column_count(entries, first, last, zero(eltype(pcv)))
        end
    end
    return nothing
end

@kernel function pmis_galerkin_rows!(
        entries, counts, @Const(slots), @Const(trp), @Const(rows), @Const(pl),
        @Const(arp), @Const(acv), @Const(prp), @Const(pcv)
    )
    i = @index(Global)
    @inbounds begin
        first = slots[i]
        last = pmis_fill_galerkin_row!(entries, first, trp, rows, pl, arp, acv, prp, pcv, i)
        # Rows are independent. Group triples by output column, with their
        # indices as tie breakers so the accumulation order is deterministic.
        pmis_heapsort_row!(entries, first, last)
        counts[i + 1] = pmis_galerkin_column_count(entries, first, last, zero(eltype(pcv)))
        i == 1 && (counts[i] = 1)
    end
end

@kernel function pmis_galerkin_pack!(
        cv, offsets, pl, ai, pr, @Const(crp), @Const(slots), @Const(entries)
    )
    i = @index(Global)
    @inbounds begin
        q = crp[i]
        prev = zero(eltype(cv))
        for t in slots[i]:(slots[i + 1] - 1)
            col, left, a, right = entries[t]
            if col != prev
                cv[q] = col
                offsets[q] = t
                q += 1
                prev = col
            end
            pl[t], ai[t], pr[t] = left, a, right
        end
        i == 1 && (offsets[length(cv) + 1] = slots[length(crp)])
    end
end

function pmis_galerkin(A, P, Pt, workspace, old, old_coarse, tracker)
    backend, block = matrix_backend(A), matrix_kernel_block_size(A)
    Tv, Ti = eltype(A), eltype(A.colval)
    nc = P.ncol
    counts = pmis_scratch!(workspace, :row_counts, backend, Int, nc + 1, tracker)
    slots = pmis_scratch!(workspace, :galerkin_slots, backend, Int, nc + 1, tracker)
    pmis_galerkin_counts!(backend, block)(
        counts, Pt.offsets, Pt.fine_rows, A.rowptr, A.colval, P.rowptr; ndrange = nc
    )
    nt = pmis_scan!(slots, counts, backend)
    entries = pmis_scratch!(workspace, :galerkin_entries, backend, NTuple{4, Ti}, nt, tracker)
    row_args = (
        entries, counts, slots, Pt.offsets, Pt.fine_rows, Pt.p_indices,
        A.rowptr, A.colval, P.rowptr, P.colval,
    )
    if backend isa KernelAbstractions.CPU
        pmis_galerkin_rows_cpu!(row_args..., nc, block)
    else
        pmis_galerkin_rows!(backend, block)(row_args...; ndrange = nc)
    end
    scan = pmis_scratch!(workspace, :row_scan, backend, Int, nc + 1, tracker)
    nz = pmis_index_limit(Ti, pmis_scan!(scan, counts, backend))
    rp = pmis_offsets(Ti, scan, backend, block, optional_property(old_coarse, :rowptr), tracker)
    cv = pmis_buffer(optional_property(old_coarse, :colval), backend, Ti, nz, tracker)
    av = pmis_buffer(optional_property(old_coarse, :nzval), backend, Tv, nz, tracker)
    offsets = pmis_buffer(optional_property(old, :offsets), backend, Int, nz + 1, tracker)
    pl = pmis_buffer(optional_property(old, :p_left), backend, Ti, nt, tracker)
    ai = pmis_buffer(optional_property(old, :a_index), backend, Ti, nt, tracker)
    pr = pmis_buffer(optional_property(old, :p_right), backend, Ti, nt, tracker)
    pmis_galerkin_pack!(backend, block)(cv, offsets, pl, ai, pr, rp, slots, entries; ndrange = nc)
    G = GalerkinMap{Ti, typeof(offsets), typeof(pl), typeof(ai), typeof(pr)}(offsets, pl, ai, pr)
    coarse = csr_matrix(rp, cv, av, nc, nc; backend, block_size = block, use_vendor_linalg = A.use_vendor_linalg)
    galerkin!(coarse, A, P, G)
    return coarse, G
end

function pmis_coarse_solver(A, options, old)
    options.coarse_solver == :spai0 && return nothing
    n = matrix_nrows(A)
    # Like BoomerAMG, use relaxation if coarsening stalled or exhausted its
    # levels before reaching the requested direct-solve size.
    n <= options.coarse_size || return nothing
    backend = matrix_backend(A)
    backend isa KernelAbstractions.CPU && return build_coarse_solver(A, backend, old)
    # coarse_size is user configurable; it must not raise the dense memory cap.
    n <= 256 || return nothing
    native_dense_lu(A.nzval) || throw(
        ArgumentError("PMIS requires native dense LU on this backend; use coarse_solver=:spai0")
    )
    if old isa DenseLUState && size(old.factorization.factors) == size(A)
        return update_coarse_solver!(old, A)
    end
    dense = KernelAbstractions.zeros(backend, eltype(A), n, n)
    copy_csr_to_dense!(dense, A)
    return DenseLUState(lu!(dense))
end

function pmis_level(A::StaticSparsityMatrixCSR{Tv, Ti}, P, Pt, G, strong, cf, cmap, options, old, tracker) where {Tv, Ti}
    backend = matrix_backend(A)
    smoother = setup_smoother(
        A, options.smoother; reuse = optional_property(old, :smoother),
        reallocation_tracker = tracker
    )
    solver = isnothing(P) ? pmis_coarse_solver(A, options, optional_property(old, :coarse_solver)) : nothing
    n = matrix_nrows(A)
    nc = isnothing(P) ? n : P.ncol
    r = pmis_buffer(optional_property(old, :residual), backend, Tv, n, tracker)
    xc = pmis_buffer(optional_property(old, :correction), backend, Tv, nc, tracker)
    rhs = pmis_buffer(optional_property(old, :rhs), backend, Tv, nc, tracker)
    return AMGLevel{Tv, Ti}(A, P, Pt, G, smoother, solver, r, xc, rhs, cf, cmap, strong, nothing)
end

function build_pmis_hierarchy(
        A::StaticSparsityMatrixCSR{Tv, Ti}, options, workspace,
        reuse_levels, reuse_finest_structure, prefix, tracker
    ) where {Tv, Ti}
    Tv <: Union{Float32, Float64} || throw(ArgumentError("PMIS requires Float32 or Float64 coefficients"))
    options.smoother isa SPAI0 || throw(ArgumentError("PMIS requires a SPAI0 smoother"))
    options.aggressive_levels == 0 || throw(ArgumentError("PMIS does not support aggressive coarsening"))
    options.interpolation isa Union{ClassicalInterpolation, ExtendedIInterpolation} ||
        throw(ArgumentError("PMIS requires Classical or Extended+i interpolation"))
    backend = matrix_backend(A)
    levels = copy(prefix)
    current = if reuse_finest_structure
        A
    else
        rp = backend_copy(backend, A.rowptr)
        cv = backend_copy(backend, A.colval)
        av = backend_copy(backend, A.nzval)
        csr_matrix(
            rp, cv, av, matrix_nrows(A), matrix_ncols(A);
            backend, block_size = options.block_size, use_vendor_linalg = A.use_vendor_linalg
        )
    end
    for l in (length(prefix) + 1):(options.max_levels - 1)
        n = matrix_nrows(current)
        n <= options.coarse_size && break
        old = reuse_level(reuse_levels, l)
        same_fine_pattern = !isnothing(old) && current === old.A
        strong = pmis_buffer(optional_property(old, :strength), backend, Bool, matrix_nonzeros(current), tracker)
        compare_strength = same_fine_pattern && !isnothing(old.P)
        buffer = if compare_strength
            pmis_scratch!(workspace, :candidate_strength, backend, Bool, length(strong), tracker)
        else
            strong
        end
        strength(current, options.coarsening.theta, options.max_row_sum, buffer; strength_type = options.strength_type)
        interpolation = options.interpolation
        cf, cmap, nc = if compare_strength && pmis_vectors_match(strong, buffer)
            old.cf, old.coarse_map, old.P.ncol
        else
            if compare_strength
                copy_kernel!(backend, matrix_kernel_block_size(current))(
                    strong, buffer, length(strong); ndrange = length(strong)
                )
            end
            pmis_split(current, strong, interpolation, workspace, old, tracker)
        end
        0 < nc < n || break
        P, same_p_pattern = pmis_prolongation(
            current, cf, cmap, nc, strong, interpolation, workspace,
            optional_property(old, :P), same_fine_pattern, tracker
        )
        old_coarse = optional_property(reuse_level(reuse_levels, l + 1), :A)
        if same_p_pattern
            Pt, G = old.Pt, old.galerkin
            coarse = galerkin!(old_coarse, current, P, G)
        else
            off, rows, idx = pmis_transpose(
                P.rowptr, P.colval, nothing, n, nc, backend, matrix_kernel_block_size(current),
                workspace, optional_property(old, :Pt), tracker
            )
            Pt = TransposeMap{Ti, typeof(off), typeof(rows), typeof(idx)}(off, rows, idx)
            coarse, G = pmis_galerkin(
                current, P, Pt, workspace, optional_property(old, :galerkin), old_coarse, tracker
            )
        end
        push!(levels, pmis_level(current, P, Pt, G, strong, cf, cmap, options, old, tracker))
        current = coarse
    end
    old = reuse_level(reuse_levels, length(levels) + 1)
    push!(levels, pmis_level(current, nothing, nothing, nothing, nothing, nothing, nothing, options, old, tracker))
    synchronize_backend(backend)
    return levels
end
