# Hypre agg_interp_type=1: ExtPI(A, C1) * PartialExtPI(A, C2).
# Removed C1 points are F points for the weight formula; P2 only stores C1 rows.
# Keeping a separate row map implements CorrectCFMarker2 without needing a
# third marker value in the shared Extended+i kernels.
function build_aggressive_prolongation(A, strong, options, interpolation::TwoStageExtendedIInterpolation)
    first_cf, first_map, first_count = cf_split(A, strong, options.coarsening)
    P1 = build_prolongation(A, first_cf, first_map, first_count, strong, interpolation.stage)
    graph, graph_strength = second_strength_graph(
        A, strong, first_cf, first_map,
        first_count, options.aggressive_num_paths
    )
    second_cf, _, _ = cf_split(graph, graph_strength, options.coarsening)
    for I in eachindex(second_cf)
        isempty(nzrange(graph, I)) && (second_cf[I] = Int8(1))
    end
    rows = findall(==(Int8(1)), first_cf)
    cf = fill(Int8(-1), length(first_cf))
    for i in rows
        second_cf[first_map[i]] == 1 && (cf[i] = Int8(1))
    end
    # Path filtering or a directed graph can leave a C1 point uncovered.
    # Repair only C1: promoting original F points would invalidate P1's columns.
    for i in rows
        cf[i] == 1 || has_interpolation_path(A, strong, cf, i, interpolation.stage) ||
            (cf[i] = Int8(1))
    end
    cmap = zeros(eltype(A.colval), length(cf))
    nc = rebuild_coarse_map!(cmap, cf)
    P2 = build_prolongation(A, cf, cmap, nc, strong, interpolation.stage; rows = rows)
    P, product, offsets, left, right, retained = interpolation_product(P1, P2, interpolation.final)
    plan = (; P1, P2, first_cf, first_map, rows, product, offsets, left, right, retained)
    return cf, cmap, nc, P, plan
end

function prolongation_from_arrays(rp, cv, values, nrow, ncol)
    return Prolongation{eltype(values), eltype(cv), typeof(rp), typeof(cv), typeof(values)}(
        rp, cv, values, nrow, ncol
    )
end

# Symbolic sparse product and its fixed numeric accumulation map. Retain the
# complete product as well as the truncated P so reuse preserves its row sum.
function interpolation_product(P1, P2, interpolation)
    Ti, Tv = eltype(P1.colval), eltype(P1.nzval)
    rp, cv, values = Ti[1], Ti[], Tv[]
    offsets, left, right = Int[1], Int[], Int[]
    out_rp, out_cv, out_values, retained = Ti[1], Ti[], Tv[], Int[]
    for i in 1:P1.nrow
        terms = Dict{Ti, Vector{Tuple{Int, Int}}}()
        for p in P1.rowptr[i]:(P1.rowptr[i + 1] - 1)
            j = P1.colval[p]
            for q in P2.rowptr[j]:(P2.rowptr[j + 1] - 1)
                push!(get!(terms, P2.colval[q], Tuple{Int, Int}[]), (Int(p), Int(q)))
            end
        end
        columns = sort!(collect(keys(terms)))
        row_values = Tv[]
        row_indices = Dict{Ti, Int}()
        for J in columns
            value = zero(Tv)
            for (p, q) in terms[J]
                push!(left, p)
                push!(right, q)
                value += P1.nzval[p] * P2.nzval[q]
            end
            push!(offsets, length(left) + 1)
            push!(cv, J)
            push!(values, value)
            push!(row_values, value)
            row_indices[J] = length(values)
        end
        push!(rp, Ti(length(cv) + 1))
        original_sum = sum(row_values)
        sort_candidates!(columns, row_values, interpolation.norm_p)
        count = candidate_count(row_values, interpolation)
        kept_sum = sum(@view row_values[1:count])
        scale = interpolation.rescale && !iszero(kept_sum) ? original_sum / kept_sum : one(Tv)
        sort_selected_columns!(columns, row_values, count)
        for k in 1:count
            push!(out_cv, columns[k])
            push!(out_values, scale * row_values[k])
            push!(retained, row_indices[columns[k]])
        end
        push!(out_rp, Ti(length(out_cv) + 1))
    end
    product = prolongation_from_arrays(rp, cv, values, P1.nrow, P2.ncol)
    P = prolongation_from_arrays(out_rp, out_cv, out_values, P1.nrow, P2.ncol)
    return P, product, offsets, left, right, retained
end

function aggressive_on_backend(plan, backend, old = nothing; reallocation_tracker = nothing)
    copy_array(key) = copy_reusing(
        optional_property(old, key), getproperty(plan, key), backend;
        reallocation_tracker = reallocation_tracker
    )
    copy_prolongation(key) = device_prolongation_reusing(
        getproperty(plan, key), backend,
        optional_property(old, key); reallocation_tracker = reallocation_tracker
    )
    return (;
        P1 = copy_prolongation(:P1), P2 = copy_prolongation(:P2),
        product = copy_prolongation(:product),
        first_cf = copy_array(:first_cf), first_map = copy_array(:first_map),
        rows = copy_array(:rows), offsets = copy_array(:offsets),
        left = copy_array(:left), right = copy_array(:right), retained = copy_array(:retained),
    )
end

@kernel function update_partial_extended_p_kernel!(
        pv, @Const(prp), @Const(pcv), @Const(rows),
        @Const(arp), @Const(acv), @Const(av),
        @Const(cf), @Const(cmap), @Const(strong), rescale, n
    )
    row = @index(Global)
    if row <= n
        i = rows[row]
        firstp, lastp = prp[row], prp[row + 1] - 1
        if firstp <= lastp
            if cf[i] == 1
                @inbounds pv[firstp] = one(eltype(pv))
            else
                kept_sum = zero(eltype(pv))
                original_sum = zero(eltype(pv))
                @inbounds for p in firstp:lastp
                    diagonal, total, selected = interpolation_row_terms(
                        arp, acv, av, cf, cmap, strong, i, pcv[p], true
                    )
                    scale = !iszero(diagonal) ? -inv(diagonal) : one(eltype(pv))
                    pv[p] = scale * selected
                    kept_sum += pv[p]
                    original_sum = scale * total
                end
                if rescale && !iszero(kept_sum)
                    scale = original_sum / kept_sum
                    @inbounds for p in firstp:lastp
                        pv[p] *= scale
                    end
                end
            end
        end
    end
end

@kernel function interpolation_product_kernel!(
        values, @Const(p1), @Const(p2),
        @Const(offsets), @Const(left), @Const(right), n
    )
    k = @index(Global)
    if k <= n
        value = zero(eltype(values))
        @inbounds for t in offsets[k]:(offsets[k + 1] - 1)
            value += p1[left[t]] * p2[right[t]]
        end
        @inbounds values[k] = value
    end
end

@kernel function truncate_product_kernel!(
        values, @Const(rp), @Const(full_values),
        @Const(full_rp), @Const(retained), rescale, n
    )
    i = @index(Global)
    if i <= n
        original_sum = zero(eltype(values))
        kept_sum = zero(eltype(values))
        @inbounds for p in full_rp[i]:(full_rp[i + 1] - 1)
            original_sum += full_values[p]
        end
        @inbounds for p in rp[i]:(rp[i + 1] - 1)
            values[p] = full_values[retained[p]]
            kept_sum += values[p]
        end
        if rescale && !iszero(kept_sum)
            scale = original_sum / kept_sum
            @inbounds for p in rp[i]:(rp[i + 1] - 1)
                values[p] *= scale
            end
        end
    end
end

function update_aggressive_prolongation!(level, interpolation::TwoStageExtendedIInterpolation)
    A, P, plan = level.A, level.P, level.aggressive
    backend, block_size = matrix_backend(A), matrix_kernel_block_size(A)
    P1, P2, product = plan.P1, plan.P2, plan.product
    update_interpolation_p_kernel!(backend, block_size)(
        P1.nzval, P1.rowptr, P1.colval, A.rowptr, A.colval, A.nzval,
        plan.first_cf, plan.first_map, level.strength, true,
        interpolation.stage.rescale, P1.nrow; ndrange = P1.nrow
    )
    update_partial_extended_p_kernel!(backend, block_size)(
        P2.nzval, P2.rowptr, P2.colval, plan.rows, A.rowptr, A.colval, A.nzval,
        level.cf, level.coarse_map, level.strength,
        interpolation.stage.rescale, P2.nrow; ndrange = P2.nrow
    )
    n = length(product.nzval)
    interpolation_product_kernel!(backend, block_size)(
        product.nzval, P1.nzval, P2.nzval,
        plan.offsets, plan.left, plan.right, n; ndrange = n
    )
    truncate_product_kernel!(backend, block_size)(
        P.nzval, P.rowptr,
        product.nzval, product.rowptr, plan.retained, interpolation.final.rescale,
        P.nrow; ndrange = P.nrow
    )
    return level
end
