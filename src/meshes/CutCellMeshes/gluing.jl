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
- `area_tol::Real = 1e-10`: Minimum area for a face to be kept.
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
    area_tol::Real = 1e-10,
    extra_out::Bool = false
)
    T = Float64

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
    bnd_a_nodes_combined = Vector{Vector{Int}}(undef, nb_a)
    for bf in 1:nb_a
        bnd_a_nodes_combined[bf] = [node_map_a[n] for n in mesh_a.boundary_faces.faces_to_nodes[bf]]
    end
    bnd_b_nodes_combined = Vector{Vector{Int}}(undef, nb_b)
    for bf in 1:nb_b
        bnd_b_nodes_combined[bf] = [node_map_b[n] for n in mesh_b.boundary_faces.faces_to_nodes[bf]]
    end

    # Find candidate pairs by centroid proximity
    # A boundary face from mesh_a can match multiple from mesh_b and vice versa
    # matched_a[bf_a] = list of bf_b indices that overlap
    matched_a = Dict{Int, Vector{Int}}()
    matched_b = Dict{Int, Vector{Int}}()

    for bf_a in 1:nb_a
        for bf_b in 1:nb_b
            if norm(bnd_centroids_a[bf_a] - bnd_centroids_b[bf_b]) < face_tol
                # Check if faces share nodes (in combined numbering)
                nodes_a = Set(bnd_a_nodes_combined[bf_a])
                nodes_b = Set(bnd_b_nodes_combined[bf_b])
                shared = intersect(nodes_a, nodes_b)
                if length(shared) >= 2  # Need at least 2 shared nodes for a face intersection
                    if !haskey(matched_a, bf_a)
                        matched_a[bf_a] = Int[]
                    end
                    push!(matched_a[bf_a], bf_b)
                    if !haskey(matched_b, bf_b)
                        matched_b[bf_b] = Int[]
                    end
                    push!(matched_b[bf_b], bf_a)
                end
            end
        end
    end

    # ----------------------------------------------------------------
    # 4. Process face intersections
    # ----------------------------------------------------------------
    # For matched boundary face pairs, compute the intersection polygon.
    # The intersection becomes a new interior face. Remaining parts of each
    # boundary face stay as boundary faces.

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

    # ---- 4c. Process boundary faces ----
    # Track which boundary faces have been fully consumed
    consumed_a = falses(nb_a)
    consumed_b = falses(nb_b)

    # For each matched pair, create interior face from shared nodes
    processed_pairs = Set{Tuple{Int,Int}}()

    for (bf_a, bf_b_list) in matched_a
        cell_a = mesh_a.boundary_faces.neighbors[bf_a]
        nodes_a_set = Set(bnd_a_nodes_combined[bf_a])

        for bf_b in bf_b_list
            pair = minmax(bf_a, bf_b)
            if pair in processed_pairs
                continue
            end
            push!(processed_pairs, pair)

            cell_b = mesh_b.boundary_faces.neighbors[bf_b]
            nodes_b_set = Set(bnd_b_nodes_combined[bf_b])
            shared_nodes = collect(intersect(nodes_a_set, nodes_b_set))

            if length(shared_nodes) >= 3
                # Order the shared nodes as a polygon
                pts = [combined_nodes[n] for n in shared_nodes]
                # Compute face normal from mesh_a boundary face
                face_a_pts = [combined_nodes[n] for n in bnd_a_nodes_combined[bf_a]]
                face_normal = _polygon_normal_from_pts(face_a_pts)
                ordered_pts = order_polygon_points(pts, face_normal)
                ordered_nodes = Int[]
                for pt in ordered_pts
                    for n in shared_nodes
                        if norm(combined_nodes[n] - pt) < tol
                            push!(ordered_nodes, n)
                            break
                        end
                    end
                end

                if length(ordered_nodes) >= 3
                    area = polygon_area([combined_nodes[n] for n in ordered_nodes])
                    if area > area_tol
                        fi = add_interior_face!(ordered_nodes, cell_a, cell_b + cell_offset_b)
                        push!(new_faces_list, fi)
                        # Check if faces are fully consumed
                        if nodes_a_set == nodes_b_set
                            consumed_a[bf_a] = true
                            consumed_b[bf_b] = true
                        elseif issubset(nodes_a_set, nodes_b_set)
                            consumed_a[bf_a] = true
                        elseif issubset(nodes_b_set, nodes_a_set)
                            consumed_b[bf_b] = true
                        end
                    end
                end
            end
        end

        # If boundary face from mesh_a still has remaining nodes (partial overlap),
        # compute the residual boundary face
        if !consumed_a[bf_a]
            # Collect all shared nodes with any matched bf_b
            all_shared = Set{Int}()
            for bf_b in bf_b_list
                nodes_b_set = Set(bnd_b_nodes_combined[bf_b])
                union!(all_shared, intersect(nodes_a_set, nodes_b_set))
            end
            # Remaining nodes: only those exclusively in mesh_a face
            remaining = setdiff(nodes_a_set, all_shared)
            # If remaining forms a valid face (with shared boundary nodes),
            # reconstruct the residual boundary polygon
            residual_nodes = _compute_residual_face(bnd_a_nodes_combined[bf_a], all_shared, remaining, combined_nodes, tol)
            if length(residual_nodes) >= 3
                area = polygon_area([combined_nodes[n] for n in residual_nodes])
                if area > area_tol
                    add_boundary_face!(residual_nodes, cell_a; old_a = bf_a)
                    consumed_a[bf_a] = true
                end
            end
            if !consumed_a[bf_a]
                # Keep original boundary face
            end
        end
    end

    # Handle partially consumed mesh_b boundary faces
    for (bf_b, bf_a_list) in matched_b
        if consumed_b[bf_b]
            continue
        end
        cell_b = mesh_b.boundary_faces.neighbors[bf_b]
        nodes_b_set = Set(bnd_b_nodes_combined[bf_b])

        all_shared = Set{Int}()
        for bf_a in bf_a_list
            nodes_a_set = Set(bnd_a_nodes_combined[bf_a])
            union!(all_shared, intersect(nodes_b_set, nodes_a_set))
        end
        remaining = setdiff(nodes_b_set, all_shared)
        residual_nodes = _compute_residual_face(bnd_b_nodes_combined[bf_b], all_shared, remaining, combined_nodes, tol)
        if length(residual_nodes) >= 3
            area = polygon_area([combined_nodes[n] for n in residual_nodes])
            if area > area_tol
                add_boundary_face!(residual_nodes, cell_b + cell_offset_b; old_b = bf_b)
                consumed_b[bf_b] = true
            end
        end
    end

    # ---- 4d. Copy unmatched / unconsumed boundary faces ----
    for bf in 1:nb_a
        if !consumed_a[bf]
            cell = mesh_a.boundary_faces.neighbors[bf]
            nodes = bnd_a_nodes_combined[bf]
            add_boundary_face!(nodes, cell; old_a = bf)
        end
    end
    for bf in 1:nb_b
        if !consumed_b[bf]
            cell = mesh_b.boundary_faces.neighbors[bf]
            nodes = bnd_b_nodes_combined[bf]
            add_boundary_face!(nodes, cell + cell_offset_b; old_b = bf)
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
    _compute_residual_face(original_nodes, shared_nodes, remaining_nodes, node_points, tol)

Compute the residual polygon from a boundary face after removing the interior
intersection region. The residual includes nodes unique to this face plus the
shared nodes on the boundary of the intersection.
"""
function _compute_residual_face(
    original_nodes::Vector{Int},
    shared_nodes::Set{Int},
    remaining_nodes::Set{Int},
    node_points::Vector{SVector{3, T}},
    tol::Real
) where T
    if isempty(remaining_nodes)
        return Int[]
    end
    # The residual face keeps the original ordering but only includes
    # nodes that are either unique to this face or on the shared boundary.
    # We walk around the original face and include nodes that are NOT
    # exclusively interior shared nodes (nodes shared but not on the boundary
    # of the intersection region).
    residual = Int[]
    for n in original_nodes
        if n in remaining_nodes || n in shared_nodes
            push!(residual, n)
        end
    end
    if length(residual) < 3
        return Int[]
    end
    # Order the residual polygon
    pts = [node_points[n] for n in residual]
    face_normal = _polygon_normal_from_pts([node_points[n] for n in original_nodes])
    ordered_pts = order_polygon_points(pts, face_normal)
    ordered_nodes = Int[]
    for pt in ordered_pts
        for n in residual
            if norm(node_points[n] - pt) < tol
                push!(ordered_nodes, n)
                break
            end
        end
    end
    return ordered_nodes
end

"""
    cut_and_displace_mesh(mesh::UnstructuredMesh{3}, plane::PlaneCut;
        constant = 0.0, slope = 0.0, side = :positive, kwargs...)

Cut a 3D mesh along a plane using `cut_mesh`, displace one side, and glue the
two sides back together. The constant part slides the shifted half along the
plane (tangentially), keeping the cut interface in contact. The slope part
tilts the shifted half perpendicular to the plane, producing a rotation about
an axis in the plane through `plane.point`.

The displacement is parametrised as

    displacement = constant · t + (x - x₀) · slope · n

where `t` is a tangent direction in the plane, `n` is the plane normal, `x` is
the projection of each node onto `t`, and `x₀` is the same projection applied
to `plane.point`.

# Keyword arguments
- `constant::Real = 0.0`: Tangential shift along the plane (keeps interface in
  contact).
- `slope::Real = 0.0`: Linear tilt perpendicular to the plane (rotation).
- `side::Symbol = :positive`: Which side to shift (`:positive` or `:negative`).
- `tol::Real = 1e-6`: Node-merge tolerance for gluing.
- `face_tol::Real = 1e-4`: Face centroid proximity tolerance.
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
    slope::Real = 0.0,
    side::Symbol = :positive,
    tol::Real = 1e-6,
    face_tol::Real = 1e-4,
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
    # Build tangent direction: pick a direction in the plane
    n = plane.normal
    ref = abs(n[1]) < 0.9 ? SVector{3, Tp}(1, 0, 0) : SVector{3, Tp}(0, 1, 0)
    tangent = normalize(cross(n, ref))
    # x0 is the projection of plane.point onto the tangent direction
    x0 = dot(plane.point, tangent)

    if side == :positive
        target_mesh = mesh_pos
    elseif side == :negative
        target_mesh = mesh_neg
    else
        throw(ArgumentError("side must be :positive or :negative, got $side"))
    end

    # Shift node points: the constant part slides along the plane (tangent
    # direction) keeping the cut interface in contact; the slope part tilts
    # perpendicular to the plane (normal direction) producing a rotation.
    shifted_nodes = copy(target_mesh.node_points)
    for i in eachindex(shifted_nodes)
        pt = shifted_nodes[i]
        x = dot(pt, tangent)
        shifted_nodes[i] = pt + constant * tangent + (x - x0) * slope * n
    end

    # Reconstruct mesh with shifted nodes
    shifted_mesh = _rebuild_mesh_with_nodes(target_mesh, shifted_nodes)

    # 5. Glue the two halves together
    if side == :positive
        glue_result = glue_mesh(shifted_mesh, mesh_neg;
            tol = tol, face_tol = face_tol, area_tol = area_tol,
            extra_out = true
        )
    else
        glue_result = glue_mesh(mesh_pos, shifted_mesh;
            tol = tol, face_tol = face_tol, area_tol = area_tol,
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
