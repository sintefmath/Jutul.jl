function merge_intervals(intervals, tol)
    T = typeof(tol)
    out = intervals
    if !isempty(intervals)
        sort!(intervals; by = first)
        out = Tuple{T, T}[]
        a, b = first(intervals)
        for (c, d) in intervals[2:end]
            if c <= b+tol
                b = max(b, d)
            else
                push!(out, (a, b))
                a, b = c, d
            end
        end
        push!(out, (a, b))
    end
    return out
end

function line_intervals(parts, origin, direction, tol; limits = (-Inf, Inf))
    T = eltype(origin)
    intervals = Tuple{T, T}[]
    for p in parts
        lo, hi = T.(limits)
        for i in eachindex(p)
            a, b = p[i], p[mod1(i+1, length(p))]
            e = b-a
            offset, slope = cross2(e, origin-a), cross2(e, direction)
            if abs(slope) <= tol*norm(e)
                if offset < -tol*norm(e)
                    hi = -Inf
                    break
                end
            elseif slope > 0
                lo = max(lo, -offset/slope)
            else
                hi = min(hi, -offset/slope)
            end
        end
        if hi-lo > tol
            push!(intervals, (lo, hi))
        end
    end
    return merge_intervals(intervals, tol)
end

function overlap_intervals(a, b, tol)
    T = typeof(tol)
    out = Tuple{T, T}[]
    for (al, ar) in a, (bl, br) in b
        l, r = max(al, bl), min(ar, br)
        if r-l > tol
            push!(out, (l, r))
        end
    end
    return merge_intervals(out, tol)
end

function clip_surface_segments(frame, segments, tol; parts = frame.parts)
    T = eltype(frame.origin)
    out = Tuple{Int, Point3{T}, Point3{T}}[]
    for (face, a, b) in segments
        pa, pb = project(frame, a), project(frame, b)
        L = norm(pb-pa)
        L > tol || continue
        direction = (pb-pa)/L
        for (lo, hi) in line_intervals(parts, pa, direction, tol; limits = (0, L))
            push!(out, (face, unproject(frame, pa+lo*direction), unproject(frame, pa+hi*direction)))
        end
    end
    return out
end

"""Positive-length overlap of collinear 3D segments, or `nothing`."""
function segment_overlap(a, b, c, d, tol)
    overlap = nothing
    L = norm(b-a)
    if L > tol
        direction = (b-a)/L
        if norm(cross(c-a, direction)) <= tol && norm(cross(d-a, direction)) <= tol
            lo, hi = minmax(dot(c-a, direction), dot(d-a, direction))
            lo, hi = max(0, lo), min(L, hi)
            if hi-lo > tol
                overlap = (a+lo*direction, a+hi*direction)
            end
        end
    end
    return overlap
end

function segment_union_measure(segments, tol)
    T = typeof(tol)
    # Triangle fans can produce duplicate or partially overlapping segments.
    groups = Vector{Tuple{Point3{T}, Point3{T}}}[]
    for (a, b) in segments
        norm(b-a) > tol || continue
        assigned = false
        for group in groups
            c, d = first(group)
            direction = normalize(d-c)
            if norm(cross(a-c, direction)) <= tol && norm(cross(b-c, direction)) <= tol
                push!(group, (a, b))
                assigned = true
                break
            end
        end
        if !assigned
            push!(groups, [(a, b)])
        end
    end
    length_sum, center = zero(T), zero(Point3{T})
    for group in groups
        origin, endpoint = first(group)
        direction = normalize(endpoint-origin)
        intervals = [minmax(dot(a-origin, direction), dot(b-origin, direction)) for (a, b) in group]
        for (lo, hi) in merge_intervals(intervals, tol)
            L = hi-lo
            length_sum += L
            center += L*(origin+(lo+hi)/2*direction)
        end
    end
    if length_sum > 0
        center /= length_sum
    end
    return length_sum, center
end
