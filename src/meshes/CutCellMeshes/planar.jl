cross2(a, b) = a[1]*b[2] - a[2]*b[1]

"""
    polygon_area_centroid(points)

Return unsigned area and the area-weighted centroid of a boundary-ordered
polygon. Signed triangle contributions preserve concave regions and avoid
cancellation from large absolute coordinates. An empty polygon returns zeros;
a zero-area polygon returns its first point as centroid.
"""
function polygon_area_centroid(p::AbstractVector{SVector{2, T}}) where T
    area = zero(T)
    centroid = zero(SVector{2, T})
    if !isempty(p)
        origin = first(p)
        for i in 2:length(p)-1
            left, right = p[i]-origin, p[i+1]-origin
            a = cross2(left, right)/2
            area += a
            centroid += a*(left+right)/3
        end
        if iszero(area)
            centroid = origin
        else
            centroid = origin+centroid/area
        end
    end
    return abs(area), centroid
end

function polygon_signed_area(p::AbstractVector{SVector{2, T}}) where T
    area = zero(T)
    if length(p) >= 3
        origin = first(p)
        for i in 2:length(p)-1
            area += cross2(p[i]-origin, p[i+1]-origin)/2
        end
    end
    return area
end

function orient_polygon(p)
    out = p
    if polygon_signed_area(p) < 0
        out = reverse(p)
    end
    return out
end

function validate_polygon(p, tol)
    n = length(p)
    for i in 1:n
        j = mod1(i+1, n)
        norm(p[i]-p[j]) > tol || throw(ArgumentError("polygon has repeated adjacent vertices"))
        for k in i+1:n
            l = mod1(k+1, n)
            if j == k || l == i
                continue
            end
            a, b, c, d = p[i], p[j], p[k], p[l]
            s1, s2 = cross2(b-a, c-a), cross2(b-a, d-a)
            s3, s4 = cross2(d-c, a-c), cross2(d-c, b-c)
            overlaps = all(min.(a, b) .<= max.(c, d) .+ tol) && all(min.(c, d) .<= max.(a, b) .+ tol)
            if overlaps && s1*s2 <= 0 && s3*s4 <= 0
                throw(ArgumentError("polygon boundary intersects itself"))
            end
        end
    end
    return nothing
end

function convex_polygon(p, tol)
    n = length(p)
    return all(cross2(p[mod1(i+1, n)]-p[i], p[mod1(i+2, n)]-p[mod1(i+1, n)]) >= -tol^2 for i in 1:n)
end

function clean_polygon(p, tol)
    out = eltype(p)[]
    for pt in p
        if isempty(out) || norm(pt-last(out)) > tol
            push!(out, pt)
        end
    end
    if length(out) > 1 && norm(first(out)-last(out)) <= tol
        pop!(out)
    end
    # Collinear vertices obstruct ear clipping and add no geometry.
    changed = true
    while changed && length(out) > 3
        changed = false
        for i in eachindex(out)
            a = out[i]-out[mod1(i-1, length(out))]
            b = out[mod1(i+1, length(out))]-out[i]
            if abs(cross2(a, b)) <= tol*max(norm(a), norm(b))
                deleteat!(out, i)
                changed = true
                break
            end
        end
    end
    return out
end

"""
    convex_parts(polygon, tolerance; resolve_winding=false)

Decompose a simple planar polygon into nonoverlapping counter-clockwise convex
parts. Concave boundaries use ear clipping. With `resolve_winding=true`, crossing
and retraced face-fan loops are resolved into exact positive-winding regions.
All coordinates retain the input floating-point type.
"""
function convex_parts(p, tol; resolve_winding = false)
    T = eltype(eltype(p))
    p = orient_polygon(clean_polygon(p, tol))
    parts = Vector{Point2{T}}[]
    winding = false
    if length(p) >= 3
        if resolve_winding
            try
                validate_polygon(p, tol)
            catch err
                err isa ArgumentError || rethrow()
                winding = true
            end
        end
        if winding
            parts = polygon_winding_parts(p, tol)
        elseif convex_polygon(p, tol)
            push!(parts, p)
        else
            ix = collect(eachindex(p))
            while length(ix) > 3
                found = false
                for j in eachindex(ix)
                    ia, ib, ic = ix[mod1(j-1, length(ix))], ix[j], ix[mod1(j+1, length(ix))]
                    a, b, c = p[ia], p[ib], p[ic]
                    if cross2(b-a, c-b) <= tol^2
                        continue
                    end
                    contains = false
                    for k in ix
                        if k == ia || k == ib || k == ic
                            continue
                        end
                        q = p[k]
                        if cross2(b-a, q-a) >= -tol^2 && cross2(c-b, q-b) >= -tol^2 && cross2(a-c, q-c) >= -tol^2
                            contains = true
                            break
                        end
                    end
                    if !contains
                        push!(parts, [a, b, c])
                        deleteat!(ix, j)
                        found = true
                        break
                    end
                end
                if !found
                    if resolve_winding
                        parts = polygon_winding_parts(p, tol)
                        winding = true
                        break
                    else
                        error("Could not triangulate planar section; check polygon or mesh geometry")
                    end
                end
            end
            if !winding
                push!(parts, p[ix])
            end
        end
    end
    return parts
end

"""
Resolve a section with crossing/retraced face-fan segments into convex regions
with positive winding. Slab endpoints include every segment crossing, so the
trapezoids are exact rather than a convex-hull approximation.
"""
function polygon_winding_parts(p, tol)
    T = eltype(eltype(p))
    events = [q[1] for q in p]
    n = length(p)
    for i in 1:n, j in i+1:n
        a, b = p[i], p[mod1(i+1, n)]
        c, d = p[j], p[mod1(j+1, n)]
        u, v = b-a, d-c
        denominator = cross2(u, v)
        if abs(denominator) <= eps(T)*norm(u)*norm(v)
            continue
        end
        t, s = cross2(c-a, v)/denominator, cross2(c-a, u)/denominator
        if 0 < t < 1 && 0 < s < 1
            push!(events, a[1]+t*u[1])
        end
    end
    sort!(events)
    xs = T[]
    for x in events
        if isempty(xs) || x-last(xs) > tol
            push!(xs, x)
        end
    end
    parts = Vector{Point2{T}}[]
    for k in 1:length(xs)-1
        xl, xr = xs[k], xs[k+1]
        xm = (xl+xr)/2
        edges = Tuple{T, T, T, Int}[]
        for i in 1:n
            a, b = p[i], p[mod1(i+1, n)]
            if min(a[1], b[1]) < xm < max(a[1], b[1])
                slope = (b[2]-a[2])/(b[1]-a[1])
                yl, yr = a[2]+slope*(xl-a[1]), a[2]+slope*(xr-a[1])
                sign = -1
                if b[1] > a[1]
                    sign = 1
                end
                push!(edges, ((yl+yr)/2, yl, yr, sign))
            end
        end
        sort!(edges; by = first)
        winding = 0
        for i in 1:length(edges)-1
            lower, upper = edges[i], edges[i+1]
            winding += lower[4]
            if winding > 0 && upper[1]-lower[1] > tol
                q = clean_polygon([Point2{T}(xl, lower[2]), Point2{T}(xr, lower[3]),
                    Point2{T}(xr, upper[3]), Point2{T}(xl, upper[2])], tol)
                if length(q) >= 3 && polygon_area_centroid(q)[1] > tol^2
                    push!(parts, q)
                end
            end
        end
    end
    return parts
end

"""
    clip_polygon(polygon, convex_boundary, tolerance=0; clean=true)

Clip a boundary-ordered convex 2D polygon against a counter-clockwise convex
boundary. Decompose concave surfaces with `convex_parts` before clipping.
`tolerance` is an absolute coordinate length. `clean=true` removes repeated
and collinear vertices; set it to false to preserve legacy gluing vertex lists.
"""
function clip_polygon(p, boundary, tol = zero(eltype(eltype(p))); clean = true)
    T = eltype(eltype(p))
    out = p
    for i in eachindex(boundary)
        out = clip_halfplane(out, boundary[i], boundary[mod1(i+1, length(boundary))], tol; clean = clean)
        if length(out) < 3
            out = Point2{T}[]
            break
        end
    end
    return out
end

function clip_halfplane(p, a, b, tol; clean = true)
    e = b-a
    normal = SVector(-e[2], e[1])
    return clip_polygon_halfspace(p, a, normal, tol; clean = clean)
end

function clip_polygon_halfspace(p, origin, normal, tol; positive = true, clean = true)
    T = eltype(normal)
    out = eltype(p)[]
    if !isempty(p)
        threshold = tol*norm(normal)
        l = last(p)
        dl = dot(normal, l-origin)
        for r in p
            dr = dot(normal, r-origin)
            if positive
                inside_l, inside_r = dl >= -threshold, dr >= -threshold
            else
                inside_l, inside_r = dl < threshold, dr < threshold
            end
            denominator = dl-dr
            if inside_l != inside_r && abs(denominator) > eps(float(T))*max(abs(dl), abs(dr))
                push!(out, l+clamp(dl/denominator, zero(T), one(T))*(r-l))
            end
            if inside_r
                push!(out, r)
            end
            l, dl = r, dr
        end
    end
    if clean
        out = clean_polygon(out, tol)
    end
    return out
end

"""
    split_polygon_by_line(polygon, origin, normal; tolerance=0, clean=false)

Return positive and negative polygons relative to a 2D oriented line. With
zero tolerance, on-line edges belong to the positive side. The default keeps
legacy gluing vertices; cleanup and an absolute length tolerance are optional.
"""
function split_polygon_by_line(p, origin, normal; tolerance = zero(eltype(normal)), clean = false)
    inside = clip_polygon_halfspace(p, origin, normal, tolerance; clean = clean)
    outside = clip_polygon_halfspace(p, origin, normal, tolerance; positive = false, clean = clean)
    return inside, outside
end

function point_index!(points, p, tol)
    index = findfirst(q -> norm(p-q) <= tol, points)
    if isnothing(index)
        push!(points, p)
        index = length(points)
    end
    return index
end

"""
    polygon_parts_area_centroid(parts, frame)

Return the total area and 3D area-weighted centroid of nonoverlapping convex
parts in `frame`'s 2D coordinates. Empty or zero-area parts return zero area
and `frame.origin` as centroid.
"""
function polygon_parts_area_centroid(parts, frame)
    T = eltype(frame.origin)
    area = zero(T)
    centroid = zero(Point2{T})
    for p in parts
        A, c = polygon_area_centroid(p)
        area += A
        centroid += A*c
    end
    if area > 0
        centroid /= area
    end
    return area, unproject(frame, centroid)
end

function connected_polygon_parts(parts, tol)
    T = typeof(tol)
    components = collect(eachindex(parts))
    for i in eachindex(parts), j in i+1:length(parts)
        component_root(components, i) == component_root(components, j) && continue
        connected = false
        for ai in eachindex(parts[i]), bi in eachindex(parts[j])
            a, b = parts[i][ai], parts[i][mod1(ai+1, length(parts[i]))]
            c, d = parts[j][bi], parts[j][mod1(bi+1, length(parts[j]))]
            norm(b-a) > tol || continue
            direction = normalize(b-a)
            if abs(cross2(c-a, direction)) <= tol && abs(cross2(d-a, direction)) <= tol
                lo, hi = minmax(dot(c-a, direction), dot(d-a, direction))
                if min(norm(b-a), hi)-max(0, lo) > tol
                    connected = true
                    break
                end
            end
        end
        if connected
            components[component_root(components, j)] = component_root(components, i)
        end
    end
    groups = Dict{Int, Vector{Vector{Point2{T}}}}()
    for i in eachindex(parts)
        push!(get!(groups, component_root(components, i), Vector{Point2{T}}[]), parts[i])
    end
    return [groups[i] for i in sort!(collect(keys(groups)))]
end

function component_root(parents, i)
    while parents[i] != i
        i = parents[i]
    end
    return i
end
