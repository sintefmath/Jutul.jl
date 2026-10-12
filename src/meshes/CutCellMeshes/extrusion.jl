"""
    polygon_prism(parts, frame, thickness, tolerance)

Extrude nonoverlapping convex 2D parts along `frame.normal`, centered on the
plane. Return `(vertices, faces)` with boundary-ordered face vertex indices
and coordinates in the frame's floating-point type. Shared edges, including
partial overlaps, are split and removed from the side walls. Caps retain
their convex decomposition, allowing concave or disconnected surfaces.
"""
function polygon_prism(parts, frame, aperture, tol)
    T = eltype(frame.origin)
    points = Point2{T}[]
    polygons = [[point_index!(points, p, tol) for p in orient_polygon(part)] for part in parts]
    caps = Vector{Int}[]
    edges = Set{Tuple{Int, Int}}()
    for polygon in polygons
        cap = Int[]
        for i in eachindex(polygon)
            a, b = polygon[i], polygon[mod1(i+1, length(polygon))]
            origin, direction = points[a], points[b]-points[a]
            distance = norm(direction)
            direction /= distance
            # Split shared edges at all patch vertices. Clipping can produce
            # partial edge overlaps between the convex parts of one cell.
            splits = Tuple{T, Int}[(0, a), (distance, b)]
            for (j, p) in enumerate(points)
                t = dot(p-origin, direction)
                if tol < t < distance-tol && abs(cross2(direction, p-origin)) <= tol
                    push!(splits, (t, j))
                end
            end
            sort!(splits; by = first)
            for k in 1:length(splits)-1
                l, r = splits[k][2], splits[k+1][2]
                push!(cap, l)
                if (r, l) in edges
                    delete!(edges, (r, l))
                else
                    push!(edges, (l, r))
                end
            end
        end
        push!(caps, cap)
    end
    n = length(points)
    lower = [unproject(frame, p)-aperture/2*frame.normal for p in points]
    upper = [unproject(frame, p)+aperture/2*frame.normal for p in points]
    faces = Vector{Int}[]
    for cap in caps
        push!(faces, reverse(cap))
        push!(faces, cap .+ n)
    end
    for (l, r) in sort!(collect(edges))
        push!(faces, [l, r, r+n, l+n])
    end
    return vcat(lower, upper), faces
end
