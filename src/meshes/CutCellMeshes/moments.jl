function polygon_average_distance(parts, origin, normal, tol)
    T = eltype(origin)
    integral = zero(T)
    area = zero(T)
    direction = Point2{T}(normal[2], -normal[1])
    for p in parts
        A, c = polygon_area_centroid(p)
        area += A
        total = A*dot(c-origin, normal)
        positive = clip_halfplane(p, origin, origin+direction, tol)
        pos = zero(T)
        if length(positive) >= 3
            Ap, cp = polygon_area_centroid(positive)
            pos = Ap*dot(cp-origin, normal)
        end
        integral += 2pos-total
    end
    average = zero(T)
    if area > 0
        average = max(integral/area, 0)
    end
    return average
end

tetra_volume(a, b, c, d) = dot(b-a, cross(c-a, d-a))/6

"""Exact mean absolute plane distance in a tetrahedron, including a plane through vertices."""
function tetra_average_distance(pts, distances)
    positive = findall(>(0), distances)
    negative = findall(<=(0), distances)
    total = sum(distances)/4
    average = abs(total)
    if !isempty(positive) && !isempty(negative)
        if length(positive) == 3
            average = tetra_average_distance(pts, -distances)
        else
            V = abs(tetra_volume(pts...))
            average = zero(total)
            if V > 0
                crossing(i, j) = pts[i]+distances[i]/(distances[i]-distances[j])*(pts[j]-pts[i])
                if length(positive) == 1
                    i = only(positive)
                    q = map(j -> crossing(i, j), negative)
                    pos = abs(tetra_volume(pts[i], q...))/V*distances[i]/4
                else
                    i, j = positive
                    k, l = negative
                    a, b = pts[i], pts[j]
                    ac, ad, bc, bd = crossing(i, k), crossing(i, l), crossing(j, k), crossing(j, l)
                    pos = abs(tetra_volume(a, b, ac, ad))*(distances[i]+distances[j])/4
                    pos += abs(tetra_volume(b, ac, ad, bd))*distances[j]/4
                    pos += abs(tetra_volume(b, ac, bc, bd))*distances[j]/4
                    pos /= V
                end
                average = max(2pos-total, 0)
            end
        end
    end
    return average
end

"""
    cell_average_distance(mesh, cell, origin, normal)

Return the volume average of absolute distance to a plane with unit `normal`.
Signed tetrahedra use the same face-center fans and orientations as Jutul's
cell geometry. The integral is exact on this representation, including planes
through cell centers or vertices. Zero-volume cells are rejected.
"""
function cell_average_distance(mesh, cell, origin, normal)
    T = eltype(origin)
    # Signed tetrahedra preserve integrals for non-convex, star-shaped cells.
    center = Point3{T}(Jutul.compute_centroid_and_measure(mesh, Cells(), cell)[1])
    volume = zero(T)
    integral = zero(T)
    for faces in (mesh.faces, mesh.boundary_faces)
        for face in faces.cells_to_faces[cell]
            nodes = faces.faces_to_nodes[face]
            isempty(nodes) && continue
            pts = Point3{T}.(mesh.node_points[nodes])
            fc = sum(pts)/length(pts)
            sign = Jutul.face_orientation_sign(faces, face, cell)
            for i in eachindex(pts)
                tet = (center, fc, pts[i], pts[mod1(i+1, length(pts))])
                V = sign*tetra_volume(tet...)
                distances = SVector{4}(map(p -> dot(p-origin, normal), tet))
                volume += V
                integral += V*tetra_average_distance(tet, distances)
            end
        end
    end
    abs(volume) > 0 || throw(ArgumentError("matrix cell $cell has zero volume"))
    return max(integral/volume, 0)
end
