"""
    signed_distance(plane, point)

Compute the signed distance from `point` to the plane defined by `plane`.
Positive values are on the side of the normal.
"""
function signed_distance(plane::PlaneCut, pt::SVector{3})
    return dot(plane.normal, pt - plane.point)
end

"""
    classify_point(plane, pt; tol)

Return +1, -1, or 0 depending on whether `pt` is on the positive side,
negative side, or on the plane.
"""
function classify_point(plane::PlaneCut, pt::SVector{3}; tol = 1e-12)
    d = signed_distance(plane, pt)
    if d > tol
        return 1
    elseif d < -tol
        return -1
    else
        return 0
    end
end

"""
    edge_plane_intersection(p1, p2, plane)

Find the intersection point of line segment (p1, p2) with a plane.
Returns the intersection point. Assumes the segment does cross the plane.
"""
function edge_plane_intersection(p1::SVector{3, T}, p2::SVector{3, T}, plane::PlaneCut) where T
    d1 = signed_distance(plane, p1)
    d2 = signed_distance(plane, p2)
    denom = d1 - d2
    if abs(denom) < eps(T)
        return (p1 + p2) / 2
    end
    t = d1 / denom
    t = clamp(t, zero(T), one(T))
    return p1 + t * (p2 - p1)
end

"""
    cell_bounding_box(mesh, cell)

Compute the axis-aligned bounding box of a cell, returns (min_pt, max_pt).
"""
function cell_bounding_box(mesh::UnstructuredMesh{3}, cell::Int)
    T = eltype(eltype(mesh.node_points))
    lo = SVector{3, T}(T(Inf), T(Inf), T(Inf))
    hi = SVector{3, T}(T(-Inf), T(-Inf), T(-Inf))
    for face in mesh.faces.cells_to_faces[cell]
        for node in mesh.faces.faces_to_nodes[face]
            pt = mesh.node_points[node]
            lo = min.(lo, pt)
            hi = max.(hi, pt)
        end
    end
    for face in mesh.boundary_faces.cells_to_faces[cell]
        for node in mesh.boundary_faces.faces_to_nodes[face]
            pt = mesh.node_points[node]
            lo = min.(lo, pt)
            hi = max.(hi, pt)
        end
    end
    return (lo, hi)
end

"""
    cell_nodes(mesh, cell)

Get all unique node indices for a cell.
"""
function cell_nodes(mesh::UnstructuredMesh{3}, cell::Int)
    nodes = Set{Int}()
    for face in mesh.faces.cells_to_faces[cell]
        for node in mesh.faces.faces_to_nodes[face]
            push!(nodes, node)
        end
    end
    for face in mesh.boundary_faces.cells_to_faces[cell]
        for node in mesh.boundary_faces.faces_to_nodes[face]
            push!(nodes, node)
        end
    end
    return nodes
end

"""
    classify_cell(mesh, cell, plane; tol)

Classify a cell with respect to a cutting plane.
Returns:
  - `:positive` if all nodes are on positive side (or on plane)
  - `:negative` if all nodes are on negative side (or on plane)
  - `:cut` if nodes span both sides
"""
function classify_cell(mesh::UnstructuredMesh{3}, cell::Int, plane::PlaneCut; tol = 1e-12)
    nodes = cell_nodes(mesh, cell)
    has_pos = false
    has_neg = false
    for n in nodes
        pt = mesh.node_points[n]
        c = classify_point(plane, pt; tol = tol)
        if c > 0
            has_pos = true
        elseif c < 0
            has_neg = true
        end
        if has_pos && has_neg
            return :cut
        end
    end
    if has_pos
        return :positive
    elseif has_neg
        return :negative
    else
        # All on plane - treat as positive
        return :positive
    end
end

"""
    clip_face_by_plane(face_nodes, node_points, plane; tol)

Clip a polygon (given by node indices) by a plane. Returns two sets of
polygon vertices: one for the positive side and one for the negative side.
Uses the Sutherland-Hodgman algorithm variant.

Returns (pos_poly, neg_poly) where each is a Vector{SVector{3, T}}.
"""
function clip_face_by_plane(
    face_nodes::AbstractVector{Int},
    node_points::Vector{SVector{3, T}},
    plane::PlaneCut;
    tol = 1e-12
) where T
    n = length(face_nodes)
    pts = [node_points[face_nodes[i]] for i in 1:n]
    dists = [signed_distance(plane, p) for p in pts]

    pos_poly = SVector{3, T}[]
    neg_poly = SVector{3, T}[]

    for i in 1:n
        j = mod1(i + 1, n)
        pi = pts[i]
        pj = pts[j]
        di = dists[i]
        dj = dists[j]

        pi_side = di > tol ? 1 : (di < -tol ? -1 : 0)
        pj_side = dj > tol ? 1 : (dj < -tol ? -1 : 0)

        # Add current point to appropriate polygon(s)
        if pi_side >= 0
            push!(pos_poly, pi)
        end
        if pi_side <= 0
            push!(neg_poly, pi)
        end

        # Check if edge crosses the plane
        if (pi_side > 0 && pj_side < 0) || (pi_side < 0 && pj_side > 0)
            inter = edge_plane_intersection(pi, pj, plane)
            push!(pos_poly, inter)
            push!(neg_poly, inter)
        end
    end
    return (pos_poly, neg_poly)
end

"""
    polygon_area(poly)

Compute the area of a 3D planar polygon.
"""
function polygon_area(poly::Vector{SVector{3, T}}) where T
    n = length(poly)
    if n < 3
        return zero(T)
    end
    c = sum(poly) / n
    area = zero(T)
    for i in 1:n
        j = mod1(i + 1, n)
        a = poly[i] - c
        b = poly[j] - c
        area += norm(cross(a, b)) / 2
    end
    return area
end

"""
    polygon_centroid(poly)

Compute the centroid of a 3D planar polygon.
"""
function polygon_centroid(poly::Vector{SVector{3, T}}) where T
    n = length(poly)
    if n == 0
        return zero(SVector{3, T})
    end
    c = sum(poly) / n
    total_area = zero(T)
    centroid = zero(SVector{3, T})
    for i in 1:n
        j = mod1(i + 1, n)
        a = poly[i] - c
        b = poly[j] - c
        tri_area = norm(cross(a, b)) / 2
        tri_centroid = (poly[i] + poly[j] + c) / 3
        total_area += tri_area
        centroid += tri_centroid * tri_area
    end
    if total_area > 0
        centroid /= total_area
    end
    return centroid
end

"""
    order_polygon_points(pts, normal)

Order polygon points counter-clockwise when viewed from the direction of `normal`.
"""
function order_polygon_points(pts::Vector{SVector{3, T}}, normal::SVector{3, T}) where T
    n = length(pts)
    if n <= 2
        return pts
    end
    c = sum(pts) / n
    # Pick two orthogonal vectors in the plane
    u = pts[1] - c
    un = norm(u)
    if un < eps(T)
        # Try another point
        for i in 2:n
            u = pts[i] - c
            un = norm(u)
            if un > eps(T)
                break
            end
        end
    end
    if un < eps(T)
        return pts  # Degenerate
    end
    u = u / un
    v = cross(normal, u)
    vn = norm(v)
    if vn < eps(T)
        return pts  # Degenerate
    end
    v = v / vn

    # Compute angles
    angles = [atan(dot(p - c, v), dot(p - c, u)) for p in pts]
    perm = sortperm(angles)
    return pts[perm]
end
