"""
    glue_mesh(mesh_a::UnstructuredMesh{3}, mesh_b::UnstructuredMesh{3}; kwargs...)

Glue two 3D `UnstructuredMesh` instances together by merging boundary faces
that are in close proximity. Nodes within distance `tol` are merged, and
overlapping boundary faces become new interior faces with the correct
neighborship. Non-overlapping boundary faces remain as boundary faces. Faces
may be split when only partially overlapping.

# Keyword arguments
- `tol::Real = 1e-6`: Distance tolerance for merging nodes.
- `face_tol::Real = 1e-4`: Tolerance for determining if two boundary face
   centroids are close enough to attempt intersection.
- `coplanar_tol::Real = 1e-3`: Maximum allowed normal-direction distance between
   two face planes for them to be considered coplanar and eligible for gluing.
- `area_tol::Real = 1e-10`: Minimum area for a face to be kept.
- `interface_point::Union{Nothing, AbstractVector} = nothing`: When provided
   together with `interface_normal`, only boundary faces whose centroids lie
   within `coplanar_tol` of the interface plane are considered for gluing.
   This prevents spurious matches between boundary faces far from the
   intended gluing interface.
- `interface_normal::Union{Nothing, AbstractVector} = nothing`: Unit normal
   of the interface plane (see `interface_point`).
- `extra_out::Bool = false`: If `true`, return `(mesh, info)` where `info` is
   a `Dict` with:
  - `"cell_index_a"` – maps new cell indices to original cell indices in
    `mesh_a` (0 for cells from `mesh_b`).
  - `"cell_index_b"` – maps new cell indices to original cell indices in
    `mesh_b` (0 for cells from `mesh_a`).
  - `"face_index_a"` – maps new interior face to original face in `mesh_a`
    (0 otherwise).
  - `"face_index_b"` – maps new interior face to original face in `mesh_b`
    (0 otherwise).
  - `"boundary_face_index_a"` – maps new boundary face to original boundary
    face in `mesh_a` (0 otherwise).
  - `"boundary_face_index_b"` – maps new boundary face to original boundary
    face in `mesh_b` (0 otherwise).
  - `"new_faces"` – indices of newly created interior faces from gluing.

Returns a new `UnstructuredMesh`, or `(UnstructuredMesh, Dict)` when
`extra_out=true`.
"""
function glue_mesh(
    mesh_a::UnstructuredMesh{3},
    mesh_b::UnstructuredMesh{3};
    tol::Real = 1e-6,
    face_tol::Real = 1e-4,
    coplanar_tol::Real = 1e-3,
    area_tol::Real = 1e-10,
    interface_point::Union{Nothing, AbstractVector} = nothing,
    interface_normal::Union{Nothing, AbstractVector} = nothing,
    extra_out::Bool = false
)
    T = Float64

    # Pre-compute interface plane filter if provided
    has_interface = !isnothing(interface_point) && !isnothing(interface_normal)
    if has_interface
        ipt = SVector{3, T}(interface_point...)
        inrm = SVector{3, T}(normalize(interface_normal)...)
    end

    nc_a = number_of_cells(mesh_a)
    nf_a = number_of_faces(mesh_a)
    nb_a = number_of_boundary_faces(mesh_a)
    nn_a = length(mesh_a.node_points)

    nc_b = number_of_cells(mesh_b)
    nf_b = number_of_faces(mesh_b)
    nb_b = number_of_boundary_faces(mesh_b)
    nn_b = length(mesh_b.node_points)

    # ----------------------------------------------------------------
    # 1. Build combined node list, merging close nodes
    # ----------------------------------------------------------------
    combined_nodes = copy(mesh_a.node_points)
    # Map from mesh_b node index → combined node index
    node_map_b = Vector{Int}(undef, nn_b)
    for nb in 1:nn_b
        pt_b = mesh_b.node_points[nb]
        merged = false
        for na in 1:length(combined_nodes)
            if norm(combined_nodes[na] - pt_b) <= tol
                node_map_b[nb] = na
                merged = true
                break
            end
        end
        if !merged
            push!(combined_nodes, pt_b)
            node_map_b[nb] = length(combined_nodes)
        end
    end
    # Identity map for mesh_a nodes
    node_map_a = collect(1:nn_a)

    # ----------------------------------------------------------------
    # 2. Cells: mesh_a cells 1..nc_a, mesh_b cells nc_a+1..nc_a+nc_b
    # ----------------------------------------------------------------
    cell_offset_b = nc_a

    # ----------------------------------------------------------------
    # 3. Find boundary face pairs that overlap
    # ----------------------------------------------------------------
    # Compute boundary face centroids for both meshes
    bnd_centroids_a = Vector{SVector{3, T}}(undef, nb_a)
    for bf in 1:nb_a
        nodes = mesh_a.boundary_faces.faces_to_nodes[bf]
        pts = [mesh_a.node_points[n] for n in nodes]
        bnd_centroids_a[bf] = sum(pts) / length(pts)
    end
    bnd_centroids_b = Vector{SVector{3, T}}(undef, nb_b)
    for bf in 1:nb_b
        nodes = mesh_b.boundary_faces.faces_to_nodes[bf]
        pts = [mesh_b.node_points[n] for n in nodes]
        bnd_centroids_b[bf] = sum(pts) / length(pts)
    end

    # For each boundary face, compute its node set in combined numbering
    # and remove consecutive duplicate nodes introduced by node merging.
    bnd_a_nodes_combined = Vector{Vector{Int}}(undef, nb_a)
    degenerate_a = falses(nb_a)
    for bf in 1:nb_a
        raw = [node_map_a[n] for n in mesh_a.boundary_faces.faces_to_nodes[bf]]
        bnd_a_nodes_combined[bf] = _deduplicate_face_nodes(raw)
        if length(unique(bnd_a_nodes_combined[bf])) < 3
            degenerate_a[bf] = true
        end
    end
    bnd_b_nodes_combined = Vector{Vector{Int}}(undef, nb_b)
    degenerate_b = falses(nb_b)
    for bf in 1:nb_b
        raw = [node_map_b[n] for n in mesh_b.boundary_faces.faces_to_nodes[bf]]
        bnd_b_nodes_combined[bf] = _deduplicate_face_nodes(raw)
        if length(unique(bnd_b_nodes_combined[bf])) < 3
            degenerate_b[bf] = true
        end
    end

    # Find candidate pairs by centroid proximity.
    # Unlike the node-sharing approach, this works even when the two meshes
    # have been displaced so that no nodes coincide.
    # Additionally, require that face normals are approximately anti-parallel
    # (facing each other), to avoid gluing exterior boundary faces that happen
    # to be close but are on the same side of the domain.
    bnd_normals_a = Vector{SVector{3, T}}(undef, nb_a)
    for bf in 1:nb_a
        pts = SVector{3,T}[mesh_a.node_points[n] for n in mesh_a.boundary_faces.faces_to_nodes[bf]]
        n_raw = _polygon_normal_from_pts(pts)
        # Orient outward: the normal should point away from the cell
        cell = mesh_a.boundary_faces.neighbors[bf]
        cell_c = _cell_centroid_from_nodes(mesh_a, cell)
        face_c = sum(pts) / length(pts)
        if dot(n_raw, face_c - cell_c) < 0
            n_raw = -n_raw
        end
        bnd_normals_a[bf] = n_raw
    end
    bnd_normals_b = Vector{SVector{3, T}}(undef, nb_b)
    for bf in 1:nb_b
        pts = SVector{3,T}[mesh_b.node_points[n] for n in mesh_b.boundary_faces.faces_to_nodes[bf]]
        n_raw = _polygon_normal_from_pts(pts)
        cell = mesh_b.boundary_faces.neighbors[bf]
        cell_c = _cell_centroid_from_nodes(mesh_b, cell)
        face_c = sum(pts) / length(pts)
        if dot(n_raw, face_c - cell_c) < 0
            n_raw = -n_raw
        end
        bnd_normals_b[bf] = n_raw
    end

    candidate_pairs = Tuple{Int,Int}[]
    for bf_a in 1:nb_a
        if degenerate_a[bf_a]
            continue
        end
        # If an interface plane is given, skip faces far from it
        if has_interface
            da = abs(dot(bnd_centroids_a[bf_a] - ipt, inrm))
            if da > coplanar_tol
                continue
            end
        end
        for bf_b in 1:nb_b
            if degenerate_b[bf_b]
                continue
            end
            if has_interface
                db = abs(dot(bnd_centroids_b[bf_b] - ipt, inrm))
                if db > coplanar_tol
                    continue
                end
            end
            if norm(bnd_centroids_a[bf_a] - bnd_centroids_b[bf_b]) < face_tol
                # Require normals to be approximately anti-parallel
                # (dot product < 0 means they face each other)
                if dot(bnd_normals_a[bf_a], bnd_normals_b[bf_b]) < 0
                    # Require faces to be approximately coplanar:
                    # the distance between centroids projected onto the face
                    # normal should be small compared to the in-plane distance.
                    n_avg = normalize(bnd_normals_a[bf_a] - bnd_normals_b[bf_b])
                    dp = bnd_centroids_a[bf_a] - bnd_centroids_b[bf_b]
                    normal_dist = abs(dot(dp, n_avg))
                    tangent_dist = norm(dp - dot(dp, n_avg) * n_avg)
                    # Accept if the normal separation is at most coplanar_tol
                    # (faces that are far apart in the normal direction are on
                    # opposite sides of the domain, not at the gluing interface)
                    if normal_dist < coplanar_tol
                        push!(candidate_pairs, (bf_a, bf_b))
                    end
                end
            end
        end
    end

    # ----------------------------------------------------------------
    # 4. Process face intersections using geometric polygon clipping
    # ----------------------------------------------------------------
    # Result accumulators
    all_face_nodes = Vector{Vector{Int}}()       # interior face node lists
    all_face_neighbors = Vector{Tuple{Int,Int}}() # interior face neighbors
    all_bnd_nodes = Vector{Vector{Int}}()          # boundary face node lists
    all_bnd_cells = Vector{Int}()                  # boundary face cells

    cell_int_faces = [Int[] for _ in 1:(nc_a + nc_b)]
    cell_bnd_faces = [Int[] for _ in 1:(nc_a + nc_b)]

    # Tracking for extra_out
    face_index_a = Int[]
    face_index_b = Int[]
    new_faces_list = Int[]
    bnd_face_index_a = Int[]
    bnd_face_index_b = Int[]

    function add_interior_face!(nodes, left, right; old_a = 0, old_b = 0)
        push!(all_face_nodes, nodes)
        push!(all_face_neighbors, (left, right))
        fi = length(all_face_nodes)
        push!(cell_int_faces[left], fi)
        push!(cell_int_faces[right], fi)
        push!(face_index_a, old_a)
        push!(face_index_b, old_b)
        return fi
    end

    function add_boundary_face!(nodes, cell; old_a = 0, old_b = 0)
        push!(all_bnd_nodes, nodes)
        push!(all_bnd_cells, cell)
        bi = length(all_bnd_nodes)
        push!(cell_bnd_faces[cell], bi)
        push!(bnd_face_index_a, old_a)
        push!(bnd_face_index_b, old_b)
        return bi
    end

    # Helper: find or add a node in combined_nodes
    function find_or_add_node!(pt)
        for i in eachindex(combined_nodes)
            if norm(combined_nodes[i] - pt) <= tol
                return i
            end
        end
        push!(combined_nodes, pt)
        return length(combined_nodes)
    end

    # ---- 4a. Copy interior faces from mesh_a ----
    for f in 1:nf_a
        l, r = mesh_a.faces.neighbors[f]
        nodes = [node_map_a[n] for n in mesh_a.faces.faces_to_nodes[f]]
        add_interior_face!(nodes, l, r; old_a = f)
    end

    # ---- 4b. Copy interior faces from mesh_b ----
    for f in 1:nf_b
        l, r = mesh_b.faces.neighbors[f]
        nodes = [node_map_b[n] for n in mesh_b.faces.faces_to_nodes[f]]
        add_interior_face!(nodes, l + cell_offset_b, r + cell_offset_b; old_b = f)
    end

    # ---- 4c. Process boundary face pairs ----
    # For each boundary face, accumulate the intersection polygons that have
    # been carved out of it.  Whatever remains becomes a residual boundary face.
    consumed_area_a = zeros(T, nb_a)  # total intersection area carved from bf_a
    consumed_area_b = zeros(T, nb_b)
    original_area_a = zeros(T, nb_a)
    original_area_b = zeros(T, nb_b)
    for bf in 1:nb_a
        original_area_a[bf] = polygon_area([combined_nodes[n] for n in bnd_a_nodes_combined[bf]])
    end
    for bf in 1:nb_b
        original_area_b[bf] = polygon_area([combined_nodes[n] for n in bnd_b_nodes_combined[bf]])
    end

    # For residual computation, track which intersection polygons were carved
    # from each boundary face.
    carved_from_a = Dict{Int, Vector{Vector{SVector{3,T}}}}()
    carved_from_b = Dict{Int, Vector{Vector{SVector{3,T}}}}()

    for (bf_a, bf_b) in candidate_pairs
        cell_a = mesh_a.boundary_faces.neighbors[bf_a]
        cell_b = mesh_b.boundary_faces.neighbors[bf_b]

        # Get 3D polygon vertices for both faces
        pts_a = SVector{3,T}[combined_nodes[n] for n in bnd_a_nodes_combined[bf_a]]
        pts_b = SVector{3,T}[combined_nodes[n] for n in bnd_b_nodes_combined[bf_b]]

        # Compute face normal from mesh_a boundary face for projection
        normal_a = _polygon_normal_from_pts(pts_a)
        face_normal = normal_a

        # Build a local 2D coordinate system on the projection plane
        centroid_ab = (sum(pts_a)/length(pts_a) + sum(pts_b)/length(pts_b)) / 2
        u, v = _build_tangent_basis(face_normal)

        # Project both polygons to 2D
        poly_a_2d = [SVector{2,T}(dot(p - centroid_ab, u), dot(p - centroid_ab, v)) for p in pts_a]
        poly_b_2d = [SVector{2,T}(dot(p - centroid_ab, u), dot(p - centroid_ab, v)) for p in pts_b]

        # Compute 2D polygon intersection via Sutherland-Hodgman
        isect_2d = _polygon_intersection_2d(poly_a_2d, poly_b_2d)

        if length(isect_2d) < 3
            continue
        end
        # Compute area of intersection
        isect_3d = [centroid_ab + p[1] * u + p[2] * v for p in isect_2d]
        isect_area = polygon_area(isect_3d)
        if isect_area <= area_tol
            continue
        end

        # Order the intersection polygon and add nodes.
        # The 3D reconstruction via centroid_ab + u,v is already at the
        # correct height: centroid_ab lies on the average of the two face
        # planes, and the coplanar_tol check guarantees those planes are
        # nearly identical, so no further projection is needed.
        isect_ordered = order_polygon_points(isect_3d, face_normal)
        face_node_ids = Int[find_or_add_node!(pt) for pt in isect_ordered]

        # Determine correct left/right orientation.
        # The face normal (from Newell on ordered nodes) should point from
        # left to right.  Compute cell centroids and check.
        face_centroid = sum(isect_ordered) / length(isect_ordered)
        cell_a_centroid = _cell_centroid_from_nodes(mesh_a, cell_a)
        cell_b_centroid = _cell_centroid_from_nodes_b(mesh_b, cell_b, combined_nodes, node_map_b)
        new_normal = _polygon_normal_from_pts(isect_ordered)

        # The normal should point from left to right. If it points away
        # from cell_a toward cell_b, then left=cell_a, right=cell_b.
        dir_a_to_b = cell_b_centroid - cell_a_centroid
        if dot(new_normal, dir_a_to_b) >= 0
            fi = add_interior_face!(face_node_ids, cell_a, cell_b + cell_offset_b)
        else
            fi = add_interior_face!(reverse(face_node_ids), cell_b + cell_offset_b, cell_a)
        end
        push!(new_faces_list, fi)

        consumed_area_a[bf_a] += isect_area
        consumed_area_b[bf_b] += isect_area

        # Track carved polygons for residual computation
        if !haskey(carved_from_a, bf_a)
            carved_from_a[bf_a] = Vector{SVector{3,T}}[]
        end
        push!(carved_from_a[bf_a], isect_3d)
        if !haskey(carved_from_b, bf_b)
            carved_from_b[bf_b] = Vector{SVector{3,T}}[]
        end
        push!(carved_from_b[bf_b], isect_3d)
    end

    # ---- 4d. Handle boundary faces ----
    # A boundary face is "fully consumed" if the intersection area covers
    # (nearly) all of its original area.  Otherwise it remains as a boundary
    # face (possibly with residual geometry, but we keep the original polygon
    # for robustness, as the residual clipping is complex).
    consumed_a = falses(nb_a)
    consumed_b = falses(nb_b)
    for bf in 1:nb_a
        if degenerate_a[bf]
            consumed_a[bf] = true
        elseif original_area_a[bf] > area_tol && consumed_area_a[bf] >= original_area_a[bf] * (1 - 1e-4)
            consumed_a[bf] = true
        end
    end
    for bf in 1:nb_b
        if degenerate_b[bf]
            consumed_b[bf] = true
        elseif original_area_b[bf] > area_tol && consumed_area_b[bf] >= original_area_b[bf] * (1 - 1e-4)
            consumed_b[bf] = true
        end
    end

    # For partially consumed faces, compute residual boundary polygons
    # Use an effective area tolerance that is never smaller than a reasonable
    # minimum to avoid keeping near-zero-area slivers from polygon subtraction.
    effective_area_tol = max(area_tol, 1e-12)
    for bf in 1:nb_a
        if consumed_a[bf]
            continue
        end
        cell = mesh_a.boundary_faces.neighbors[bf]
        if haskey(carved_from_a, bf)
            residuals = _compute_residual_polygons(
                SVector{3,T}[combined_nodes[n] for n in bnd_a_nodes_combined[bf]],
                carved_from_a[bf],
                effective_area_tol
            )
            if !isempty(residuals)
                for rpoly in residuals
                    rnodes = _deduplicate_face_nodes(Int[find_or_add_node!(pt) for pt in rpoly])
                    if length(unique(rnodes)) >= 3
                        add_boundary_face!(rnodes, cell; old_a = bf)
                    end
                end
                consumed_a[bf] = true
            end
        end
    end
    for bf in 1:nb_b
        if consumed_b[bf]
            continue
        end
        cell = mesh_b.boundary_faces.neighbors[bf]
        if haskey(carved_from_b, bf)
            residuals = _compute_residual_polygons(
                SVector{3,T}[combined_nodes[n] for n in bnd_b_nodes_combined[bf]],
                carved_from_b[bf],
                effective_area_tol
            )
            if !isempty(residuals)
                for rpoly in residuals
                    rnodes = _deduplicate_face_nodes(Int[find_or_add_node!(pt) for pt in rpoly])
                    if length(unique(rnodes)) >= 3
                        add_boundary_face!(rnodes, cell + cell_offset_b; old_b = bf)
                    end
                end
                consumed_b[bf] = true
            end
        end
    end

    # Copy unmatched / unconsumed boundary faces, skipping near-zero-area faces
    for bf in 1:nb_a
        if !consumed_a[bf]
            nodes = bnd_a_nodes_combined[bf]
            face_area = polygon_area([combined_nodes[n] for n in nodes])
            if face_area > effective_area_tol
                cell = mesh_a.boundary_faces.neighbors[bf]
                add_boundary_face!(nodes, cell; old_a = bf)
            end
        end
    end
    for bf in 1:nb_b
        if !consumed_b[bf]
            nodes = bnd_b_nodes_combined[bf]
            face_area = polygon_area([combined_nodes[n] for n in nodes])
            if face_area > effective_area_tol
                cell = mesh_b.boundary_faces.neighbors[bf]
                add_boundary_face!(nodes, cell + cell_offset_b; old_b = bf)
            end
        end
    end

    # ----------------------------------------------------------------
    # 5. Build the final mesh
    # ----------------------------------------------------------------
    new_cell_count = nc_a + nc_b

    faces_nodes = Int[]
    faces_nodespos = Int[1]
    for fn in all_face_nodes
        append!(faces_nodes, fn)
        push!(faces_nodespos, faces_nodespos[end] + length(fn))
    end

    bnd_faces_nodes = Int[]
    bnd_faces_nodespos = Int[1]
    for fn in all_bnd_nodes
        append!(bnd_faces_nodes, fn)
        push!(bnd_faces_nodespos, bnd_faces_nodespos[end] + length(fn))
    end

    cells_faces = Int[]
    cells_facepos = Int[1]
    for c in 1:new_cell_count
        append!(cells_faces, cell_int_faces[c])
        push!(cells_facepos, cells_facepos[end] + length(cell_int_faces[c]))
    end

    bnd_cells_faces = Int[]
    bnd_cells_facepos = Int[1]
    for c in 1:new_cell_count
        append!(bnd_cells_faces, cell_bnd_faces[c])
        push!(bnd_cells_facepos, bnd_cells_facepos[end] + length(cell_bnd_faces[c]))
    end

    new_mesh = UnstructuredMesh(
        cells_faces,
        cells_facepos,
        bnd_cells_faces,
        bnd_cells_facepos,
        faces_nodes,
        faces_nodespos,
        bnd_faces_nodes,
        bnd_faces_nodespos,
        combined_nodes,
        all_face_neighbors,
        all_bnd_cells
    )

    if extra_out
        cell_index_a = zeros(Int, new_cell_count)
        cell_index_b = zeros(Int, new_cell_count)
        for c in 1:nc_a
            cell_index_a[c] = c
        end
        for c in 1:nc_b
            cell_index_b[nc_a + c] = c
        end

        info = Dict{String, Any}(
            "cell_index_a"          => cell_index_a,
            "cell_index_b"          => cell_index_b,
            "face_index_a"          => face_index_a,
            "face_index_b"          => face_index_b,
            "boundary_face_index_a" => bnd_face_index_a,
            "boundary_face_index_b" => bnd_face_index_b,
            "new_faces"             => new_faces_list
        )
        return (new_mesh, info)
    end
    return new_mesh
end

"""
    _polygon_normal_from_pts(pts)

Compute unit normal of a planar polygon from 3D points using Newell's method.
"""
function _polygon_normal_from_pts(pts::AbstractVector{SVector{3, T}}) where T
    n = zero(SVector{3, T})
    np = length(pts)
    for i in 1:np
        j = mod1(i + 1, np)
        a = pts[i]
        b = pts[j]
        n += cross(a, b)
    end
    nn = norm(n)
    if nn < eps(T)
        return SVector{3, T}(0, 0, 1)
    end
    return n / nn
end

"""
    _build_tangent_basis(normal)

Build an orthonormal (u, v) basis in the plane perpendicular to `normal`.
"""
function _build_tangent_basis(normal::SVector{3, T}) where T
    ref = abs(normal[1]) < T(0.9) ? SVector{3,T}(1, 0, 0) : SVector{3,T}(0, 1, 0)
    u = normalize(cross(normal, ref))
    v = cross(normal, u)
    return (u, v)
end

"""
    _ensure_ccw_2d(poly)

Ensure a 2D polygon has counter-clockwise winding order.
Uses the signed area (shoelace formula).
"""
function _ensure_ccw_2d(poly::Vector{SVector{2, T}}) where T
    n = length(poly)
    if n < 3
        return poly
    end
    # Signed area: positive for CCW, negative for CW
    area2 = zero(T)
    for i in 1:n
        j = mod1(i + 1, n)
        area2 += poly[i][1] * poly[j][2] - poly[j][1] * poly[i][2]
    end
    if area2 < 0
        return reverse(poly)
    end
    return poly
end

"""
    _polygon_intersection_2d(subject, clip)

Compute the intersection of two 2D convex polygons using the
Sutherland-Hodgman algorithm. Both polygons are reoriented to
counter-clockwise winding before clipping. Returns a vector of 2D vertices.
"""
function _polygon_intersection_2d(
    subject::Vector{SVector{2, T}},
    clip::Vector{SVector{2, T}}
) where T
    output = _ensure_ccw_2d(subject)
    clip_ccw = _ensure_ccw_2d(clip)
    nc = length(clip_ccw)
    for i in 1:nc
        if isempty(output)
            return SVector{2,T}[]
        end
        input = output
        output = SVector{2,T}[]
        edge_start = clip_ccw[i]
        edge_end = clip_ccw[mod1(i + 1, nc)]
        # Edge direction and inward normal
        edge_dir = edge_end - edge_start
        # Inward normal (pointing into the clip polygon)
        inward = SVector{2,T}(-edge_dir[2], edge_dir[1])

        n_in = length(input)
        for j in 1:n_in
            current = input[j]
            next = input[mod1(j + 1, n_in)]
            d_curr = dot(current - edge_start, inward)
            d_next = dot(next - edge_start, inward)
            if d_curr >= 0  # current inside
                push!(output, current)
                if d_next < 0  # next outside
                    # Compute intersection
                    t = d_curr / (d_curr - d_next)
                    push!(output, current + t * (next - current))
                end
            elseif d_next >= 0  # current outside, next inside
                t = d_curr / (d_curr - d_next)
                push!(output, current + t * (next - current))
            end
        end
    end
    return output
end

"""
    _compute_residual_polygons(face_poly, carved_polys, area_tol)

Compute the residual boundary polygon(s) after subtracting intersection regions
from a face polygon. For each carved polygon, clips the remaining pieces against
the complement of the carved polygon. Returns a list of residual convex polygons,
or empty if the face is fully consumed.
"""
function _compute_residual_polygons(
    face_poly::Vector{SVector{3, T}},
    carved_polys::Vector{Vector{SVector{3, T}}},
    area_tol::Real
) where T
    face_area = polygon_area(face_poly)
    carved_area = sum(polygon_area(cp) for cp in carved_polys)
    if carved_area >= face_area * (1 - 1e-4)
        return Vector{SVector{3,T}}[]
    end

    # Project everything to 2D for clipping
    face_normal = _polygon_normal_from_pts(face_poly)
    centroid = sum(face_poly) / length(face_poly)
    u, v = _build_tangent_basis(face_normal)

    face_2d = [SVector{2,T}(dot(p - centroid, u), dot(p - centroid, v)) for p in face_poly]
    face_2d = _ensure_ccw_2d(face_2d)

    # Start with the full face as the set of residual pieces
    pieces = [face_2d]

    for carved in carved_polys
        carved_2d = [SVector{2,T}(dot(p - centroid, u), dot(p - centroid, v)) for p in carved]
        carved_2d = _ensure_ccw_2d(carved_2d)

        new_pieces = Vector{SVector{2,T}}[]
        for piece in pieces
            # Subtract carved_2d from piece by clipping against each edge's exterior
            remaining = _subtract_convex_2d(piece, carved_2d)
            for r in remaining
                if _polygon_area_2d(r) > area_tol
                    push!(new_pieces, r)
                end
            end
        end
        pieces = new_pieces
    end

    # Back-project to 3D
    result = Vector{SVector{3,T}}[]
    # Use the average normal-distance of the original face
    d_face = sum(dot(p, face_normal) for p in face_poly) / length(face_poly)
    for piece in pieces
        poly_3d = SVector{3,T}[]
        for p2 in piece
            p3 = centroid + p2[1] * u + p2[2] * v
            d_pt = dot(p3, face_normal)
            p3 = p3 + (d_face - d_pt) * face_normal
            push!(poly_3d, p3)
        end
        push!(result, poly_3d)
    end
    return result
end

"""
    _subtract_convex_2d(subject, clip)

Subtract convex polygon `clip` from convex polygon `subject`.
Returns a list of convex polygon pieces that cover subject \\ clip.
Uses successive clipping against the complement of each edge of clip.
"""
function _subtract_convex_2d(
    subject::Vector{SVector{2, T}},
    clip::Vector{SVector{2, T}}
) where T
    nc = length(clip)
    # Successively subtract: we clip the subject against the OUTSIDE of each
    # edge of clip.  After subtracting the inside of edge i, any remaining
    # piece that's inside edge i+1..n needs further subtraction.
    # pieces_to_process = subject (fully inside of no edges yet removed)
    # For each edge, split each piece into (inside edge, outside edge).
    # "outside edge" pieces go directly to the result (they're outside clip).
    # "inside edge" pieces continue to the next edge for further processing.
    # After all edges, any remaining "inside all edges" pieces are fully
    # inside clip and are discarded.
    pieces = [subject]
    result = Vector{SVector{2,T}}[]

    for i in 1:nc
        j = mod1(i + 1, nc)
        edge_start = clip[i]
        edge_end = clip[j]
        edge_dir = edge_end - edge_start
        # Inward normal (pointing into the clip polygon for CCW winding)
        inward = SVector{2,T}(-edge_dir[2], edge_dir[1])

        new_pieces = Vector{SVector{2,T}}[]
        for piece in pieces
            inside, outside = _split_polygon_by_line_2d(piece, edge_start, inward)
            # "outside" pieces are outside this edge → they're outside clip → result
            if length(outside) >= 3
                push!(result, outside)
            end
            # "inside" pieces need to be checked against remaining edges
            if length(inside) >= 3
                push!(new_pieces, inside)
            end
        end
        pieces = new_pieces
    end
    # Remaining pieces are inside ALL edges → inside clip → discard
    return result
end

"""
    _split_polygon_by_line_2d(poly, point_on_line, inward_normal)

Split a 2D polygon by a line defined by a point and an inward normal.
Returns (inside, outside) where inside is the part on the inward side
and outside is the part on the outward side.
"""
function _split_polygon_by_line_2d(
    poly::Vector{SVector{2, T}},
    point_on_line::SVector{2, T},
    inward_normal::SVector{2, T}
) where T
    n = length(poly)
    inside = SVector{2,T}[]
    outside = SVector{2,T}[]

    for i in 1:n
        j = mod1(i + 1, n)
        pi = poly[i]
        pj = poly[j]
        di = dot(pi - point_on_line, inward_normal)
        dj = dot(pj - point_on_line, inward_normal)

        if di >= 0  # pi is inside
            push!(inside, pi)
            if dj < 0  # pj is outside → crossing
                t = di / (di - dj)
                inter = pi + t * (pj - pi)
                push!(inside, inter)
                push!(outside, inter)
            end
        else  # pi is outside
            push!(outside, pi)
            if dj >= 0  # pj is inside → crossing
                t = di / (di - dj)
                inter = pi + t * (pj - pi)
                push!(outside, inter)
                push!(inside, inter)
            end
        end
    end
    return (inside, outside)
end

"""
    _deduplicate_face_nodes(nodes)

Remove consecutive duplicate node indices from a face node list, including
wrap-around (last == first). Returns a new vector. Faces with fewer than 3
unique nodes after deduplication are degenerate.
"""
function _deduplicate_face_nodes(nodes::Vector{Int})
    n = length(nodes)
    if n == 0
        return nodes
    end
    result = Int[nodes[1]]
    for i in 2:n
        if nodes[i] != result[end]
            push!(result, nodes[i])
        end
    end
    # Check wrap-around: if last == first, remove the last
    while length(result) > 1 && result[end] == result[1]
        pop!(result)
    end
    return result
end

"""
    _polygon_area_2d(poly)

Compute the area of a 2D polygon using the shoelace formula.
"""
function _polygon_area_2d(poly::Vector{SVector{2, T}}) where T
    n = length(poly)
    if n < 3
        return zero(T)
    end
    area2 = zero(T)
    for i in 1:n
        j = mod1(i + 1, n)
        area2 += poly[i][1] * poly[j][2] - poly[j][1] * poly[i][2]
    end
    return abs(area2) / 2
end

"""
    _cell_centroid_from_nodes(mesh, cell)

Approximate centroid of a cell from all its unique node coordinates.
"""
function _cell_centroid_from_nodes(mesh::UnstructuredMesh{3}, cell::Int)
    nodes = cell_nodes(mesh, cell)
    T = eltype(eltype(mesh.node_points))
    c = zero(SVector{3, T})
    for n in nodes
        c += mesh.node_points[n]
    end
    return c / length(nodes)
end

"""
    _cell_centroid_from_nodes_b(mesh_b, cell_b, combined_nodes, node_map_b)

Approximate centroid of a cell from mesh_b using combined node coordinates.
"""
function _cell_centroid_from_nodes_b(mesh_b::UnstructuredMesh{3}, cell_b::Int,
                                     combined_nodes, node_map_b)
    nodes = cell_nodes(mesh_b, cell_b)
    T = eltype(eltype(combined_nodes))
    c = zero(SVector{3, T})
    for n in nodes
        c += combined_nodes[node_map_b[n]]
    end
    return c / length(nodes)
end

"""
    cut_and_displace_mesh(mesh::UnstructuredMesh{3}, plane::PlaneCut;
        constant = 0.0, shift_lr = 0.0, angle = 0.0,
        side = :positive, kwargs...)

Cut a 3D mesh along a plane using `cut_mesh`, displace one side, and glue the
two sides back together.  All three displacement operations act **in the plane**,
so the cut interface remains in contact (no gap, no collision).

Two in-plane tangent directions are defined automatically:

    t₁ = normalize(cross(n, ref))
    t₂ = cross(n, t₁)

where `n` is the plane normal and `ref` is a reference vector chosen to avoid
degeneracy.  `t₁` and `t₂` are orthogonal tangent directions within the cutting
plane.

The displacement applied to each node is the combination of:

1. A uniform shift along `t₁` (up-down):  `constant · t₁`
2. A uniform shift along `t₂` (left-right):  `shift_lr · t₂`
3. An in-plane rotation by `angle` (radians) around the plane normal `n`,
   pivoting at `plane.point`.

All three operations preserve the cut-surface contact (they slide points within
the plane but never move them out of it) and preserve cell volumes.

# Keyword arguments
- `constant::Real = 0.0`: Uniform shift along `t₁` (up-down direction).
- `shift_lr::Real = 0.0`: Uniform shift along `t₂` (left-right direction).
- `angle::Real = 0.0`: In-plane rotation (radians) around the plane normal `n`,
  pivoting at `plane.point`.
- `side::Symbol = :positive`: Which side to shift (`:positive` or `:negative`).
- `tol::Real = 1e-6`: Node-merge tolerance for gluing.
- `face_tol::Real = 1e-4`: Face centroid proximity tolerance.
- `coplanar_tol::Real = 1e-3`: Normal-direction coplanarity tolerance for face matching.
- `area_tol::Real = 1e-10`: Minimum face area to keep.
- `min_cut_fraction::Real = 0.05`: Passed to `cut_mesh`.
- `extra_out::Bool = false`: Return mapping information.

When `extra_out=true` the returned `Dict` contains:
- `"cell_index"` – maps each new cell to its original cell index in the input
  `mesh` (composed through `cut_mesh` and `extract_submesh`).
- `"cell_side"` – per cell, `:positive` or `:negative` indicating which side
  of the cut plane it belongs to.
- `"face_index_a"`, `"face_index_b"` – per interior face, index in the
  positive-side or negative-side sub-mesh (0 otherwise).
- `"boundary_face_index_a"`, `"boundary_face_index_b"` – same for boundary
  faces.
- `"new_faces"` – indices of newly created interior faces from gluing.

Returns a new `UnstructuredMesh`, or `(UnstructuredMesh, Dict)` when
`extra_out=true`.
"""
function cut_and_displace_mesh(
    mesh::UnstructuredMesh{3},
    plane::PlaneCut{Tp};
    constant::Real = 0.0,
    shift_lr::Real = 0.0,
    angle::Real = 0.0,
    side::Symbol = :positive,
    tol::Real = 1e-6,
    face_tol::Real = 1e-4,
    coplanar_tol::Real = 1e-3,
    area_tol::Real = 1e-10,
    min_cut_fraction::Real = 0.05,
    extra_out::Bool = false
) where Tp
    # 1. Cut the mesh
    cut_result = cut_mesh(mesh, plane;
        min_cut_fraction = min_cut_fraction,
        extra_out = true
    )
    cut, cut_info = cut_result

    nc_cut = number_of_cells(cut)

    # 2. Classify cells to positive / negative side
    pos_cells = Int[]
    neg_cells = Int[]
    for c in 1:nc_cut
        cl = classify_cell(cut, c, plane; tol = tol)
        if cl == :positive
            push!(pos_cells, c)
        else
            push!(neg_cells, c)
        end
    end

    # 3. Extract submeshes
    mesh_pos = extract_submesh(cut, pos_cells)
    mesh_neg = extract_submesh(cut, neg_cells)

    # 4. Apply displacement to one side
    # Build two orthogonal tangent directions in the plane
    n = plane.normal
    ref = abs(n[1]) < 0.9 ? SVector{3, Tp}(1, 0, 0) : SVector{3, Tp}(0, 1, 0)
    t1 = normalize(cross(n, ref))
    t2 = cross(n, t1)

    if side == :positive
        target_mesh = mesh_pos
    elseif side == :negative
        target_mesh = mesh_neg
    else
        throw(ArgumentError("side must be :positive or :negative, got $side"))
    end

    # Shift node points — all operations are in-plane, keeping the cut
    # interface in contact.
    # constant: uniform slide along t1 (up-down).
    # shift_lr: uniform slide along t2 (left-right).
    # angle:    in-plane rotation around n, pivoting at plane.point.
    #           For a point with in-plane coordinates (x1, x2) relative to
    #           plane.point, the rotated position is
    #             x1' = x1·cos(θ) - x2·sin(θ)
    #             x2' = x1·sin(θ) + x2·cos(θ)
    #           The out-of-plane component d is unchanged.
    cosθ = cos(angle)
    sinθ = sin(angle)
    shifted_nodes = copy(target_mesh.node_points)
    for i in eachindex(shifted_nodes)
        pt = shifted_nodes[i]
        dp = pt - plane.point
        x1 = dot(dp, t1)  # in-plane coordinate along t1
        x2 = dot(dp, t2)  # in-plane coordinate along t2
        d  = dot(dp, n)   # out-of-plane (normal) distance
        # Rotate in-plane, then shift
        x1_new = x1 * cosθ - x2 * sinθ + constant
        x2_new = x1 * sinθ + x2 * cosθ + shift_lr
        # Reconstruct the point
        shifted_nodes[i] = plane.point + x1_new * t1 + x2_new * t2 + d * n
    end

    # Reconstruct mesh with shifted nodes
    shifted_mesh = _rebuild_mesh_with_nodes(target_mesh, shifted_nodes)

    # 5. Glue the two halves together
    # Pass the cut plane info so glue_mesh only matches faces near the interface
    n_hat = normalize(plane.normal)
    if side == :positive
        glue_result = glue_mesh(shifted_mesh, mesh_neg;
            tol = tol, face_tol = face_tol, coplanar_tol = coplanar_tol,
            area_tol = area_tol,
            interface_point = plane.point,
            interface_normal = n_hat,
            extra_out = true
        )
    else
        glue_result = glue_mesh(mesh_pos, shifted_mesh;
            tol = tol, face_tol = face_tol, coplanar_tol = coplanar_tol,
            area_tol = area_tol,
            interface_point = plane.point,
            interface_normal = n_hat,
            extra_out = true
        )
    end
    glued_mesh, glue_info = glue_result

    if extra_out
        nc_a = number_of_cells(side == :positive ? shifted_mesh : mesh_pos)

        # Compose mappings back to the original mesh:
        # cut_info["cell_index"] maps cut-mesh cell → original cell.
        # pos_cells / neg_cells map sub-mesh cells → cut-mesh cell.
        nc_glued = number_of_cells(glued_mesh)
        cell_index = Vector{Int}(undef, nc_glued)
        cell_side = Vector{Symbol}(undef, nc_glued)
        for c in 1:nc_glued
            ia = glue_info["cell_index_a"][c]
            ib = glue_info["cell_index_b"][c]
            if ia > 0
                # This cell comes from mesh_a in the glue.
                # mesh_a is always the positive-side submesh (either shifted
                # or unshifted), so it maps through pos_cells.
                cut_cell = pos_cells[ia]
                cell_side[c] = :positive
            else
                # This cell comes from mesh_b in the glue.
                # mesh_b is always the negative-side submesh.
                cut_cell = neg_cells[ib]
                cell_side[c] = :negative
            end
            cell_index[c] = cut_info["cell_index"][cut_cell]
        end

        info = Dict{String, Any}(
            "cell_index"            => cell_index,
            "cell_side"             => cell_side,
            "face_index_a"          => glue_info["face_index_a"],
            "face_index_b"          => glue_info["face_index_b"],
            "boundary_face_index_a" => glue_info["boundary_face_index_a"],
            "boundary_face_index_b" => glue_info["boundary_face_index_b"],
            "new_faces"             => glue_info["new_faces"]
        )
        return (glued_mesh, info)
    end
    return glued_mesh
end

"""
    _rebuild_mesh_with_nodes(mesh, new_nodes)

Create a new UnstructuredMesh identical to `mesh` but with `new_nodes` as node
points.
"""
function _rebuild_mesh_with_nodes(
    mesh::UnstructuredMesh{3},
    new_nodes::Vector{SVector{3, T}}
) where T
    nc = number_of_cells(mesh)
    nf = number_of_faces(mesh)
    nb = number_of_boundary_faces(mesh)

    # Extract flat arrays from the mesh IndirectionMaps
    c2f = mesh.faces.cells_to_faces
    cells_faces = copy(c2f.vals)
    cells_facepos = copy(c2f.pos)

    c2b = mesh.boundary_faces.cells_to_faces
    bnd_cells_faces = copy(c2b.vals)
    bnd_cells_facepos = copy(c2b.pos)

    f2n = mesh.faces.faces_to_nodes
    faces_nodes = copy(f2n.vals)
    faces_nodespos = copy(f2n.pos)

    b2n = mesh.boundary_faces.faces_to_nodes
    bnd_faces_nodes = copy(b2n.vals)
    bnd_faces_nodespos = copy(b2n.pos)

    int_neighbors = copy(mesh.faces.neighbors)
    bnd_cells = copy(mesh.boundary_faces.neighbors)

    return UnstructuredMesh(
        cells_faces,
        cells_facepos,
        bnd_cells_faces,
        bnd_cells_facepos,
        faces_nodes,
        faces_nodespos,
        bnd_faces_nodes,
        bnd_faces_nodespos,
        new_nodes,
        int_neighbors,
        bnd_cells
    )
end
