function triangle_plane_segment(a, b, c, plane::PlaneCut, tol)
    T = eltype(plane.point)
    pts = Point3{T}[]
    vertices = (a, b, c)
    distances = map(p -> signed_distance(plane, p), vertices)
    for i in 1:3
        j = mod1(i+1, 3)
        p, q = vertices[i], vertices[j]
        dp, dq = distances[i], distances[j]
        if abs(dp) <= tol
            point_index!(pts, p-dp*plane.normal, tol)
        end
        if (dp < -tol && dq > tol) || (dq < -tol && dp > tol)
            point_index!(pts, edge_plane_intersection(p, q, plane), tol)
        end
    end
    segment = nothing
    if length(pts) == 2 && norm(pts[1]-pts[2]) > tol
        segment = (pts[1], pts[2])
    end
    return segment
end

"""
    cell_section(mesh, cell, origin, normal, u, v, tolerance)

Intersect a 3D cell with a plane using Jutul's face-center triangle fans,
including warped faces. `normal` is a unit normal and `(u, v)` are orthonormal
plane axes. Return `(loops, face_segments)`: boundary loops in local 2D
coordinates and `(interior_face_id, start, stop)` segments in 3D. Coplanar
faces take precedence over other section loops; point/edge tangencies have no
area. Apply `convex_parts(...; resolve_winding=true)` before clipping loops.
"""
function cell_section(mesh, cell, origin, normal, u, v, tol)
    T = eltype(origin)
    plane = PlaneCut(origin, normal)
    points = Point3{T}[]
    edges = Set{Tuple{Int, Int}}()
    face_segments = Tuple{Int, Point3{T}, Point3{T}}[]
    coplanar = Vector{Point2{T}}[]
    for faces in (mesh.faces, mesh.boundary_faces)
        for face in faces.cells_to_faces[cell]
            nodes = faces.faces_to_nodes[face]
            isempty(nodes) && continue # Non-neighboring connection without geometry.
            pts = Point3{T}.(mesh.node_points[nodes])
            center = sum(pts)/length(pts)
            face_id = 0
            if faces === mesh.faces
                face_id = face
            end
            if all(p -> abs(dot(p-origin, normal)) <= tol, pts)
                push!(coplanar, [Point2{T}(dot(p-origin, u), dot(p-origin, v)) for p in pts])
                continue
            end
            for i in eachindex(pts)
                seg = triangle_plane_segment(center, pts[i], pts[mod1(i+1, length(pts))], plane, tol)
                isnothing(seg) && continue
                a, b = seg
                ia, ib = point_index!(points, a, tol), point_index!(points, b, tol)
                push!(edges, minmax(ia, ib))
                if face_id > 0
                    push!(face_segments, (face_id, a, b))
                end
            end
        end
    end
    loops = coplanar
    if isempty(coplanar)
        loops = Vector{Point2{T}}[]
    end
    while isempty(coplanar) && !isempty(edges)
        a, b = pop!(edges)
        ids = [a, b]
        current = b
        while current != a
            next_edge = nothing
            for e in edges
                if current in e
                    next_edge = e
                    break
                end
            end
            if isnothing(next_edge)
                # Tangency at a vertex or an edge has no fracture area.
                empty!(ids)
                break
            end
            delete!(edges, next_edge)
            l, r = next_edge
            if l == current
                current = r
            else
                current = l
            end
            if current != a
                push!(ids, current)
            end
        end
        if length(ids) >= 3
            push!(loops, [Point2{T}(dot(points[i]-origin, u), dot(points[i]-origin, v)) for i in ids])
        end
    end
    return loops, face_segments
end

function coincident_faces(mesh, cell, frame, tol)
    out = Int[]
    for face in mesh.faces.cells_to_faces[cell]
        nodes = mesh.faces.faces_to_nodes[face]
        if !isempty(nodes) && all(node -> abs(dot(mesh.node_points[node]-frame.origin, frame.normal)) <= tol, nodes)
            push!(out, face)
        end
    end
    return out
end
