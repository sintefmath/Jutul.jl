"""
    embed_mesh(mesh_a::UnstructuredMesh{3}, mesh_b::UnstructuredMesh{3}; kwargs...)

Embed mesh B inside mesh A by removing overlapping cells from A, cutting
partially overlapping cells, and gluing the two meshes together.

The structure of mesh B is preserved exactly.  Cells in A that are entirely
inside B are removed.  Cells in A that are partially cut by the boundary of B
are split so that only the portion outside B remains.  The trimmed A and B are
then stitched together with `glue_mesh`.

# Keyword arguments
- `extra_out::Bool = false`: If `true`, return `(mesh, info)` where `info` is a
  `Dict{String, Any}` containing:
  - `"cell_origin"`:  `Vector{Symbol}` — `:mesh_a` or `:mesh_b` for each cell.
  - `"cell_index_a"`: `Vector{Int}` — maps each cell to its index in the original
    mesh A (0 for cells from B).
  - `"cell_index_b"`: `Vector{Int}` — maps each cell to its index in the original
    mesh B (0 for cells from A).
- `min_cut_fraction::Real = 0.01`: Passed to `cut_mesh` for volume filtering.
- `tol::Real = 1e-6`: Node merging tolerance for gluing.
- `face_tol::Real = 1e-4`: Face centroid proximity tolerance for gluing.
- `coplanar_tol::Real = 1e-3`: Coplanarity tolerance for gluing.
- `merge_faces::Bool = true`: Whether to merge coplanar faces after cutting.

Returns a new `UnstructuredMesh{3}`, or `(UnstructuredMesh{3}, Dict)` when
`extra_out=true`.
"""
function embed_mesh(
    mesh_a::UnstructuredMesh{3},
    mesh_b::UnstructuredMesh{3};
    extra_out::Bool = false,
    min_cut_fraction::Real = 0.01,
    tol::Real = 1e-6,
    face_tol::Real = 1e-4,
    coplanar_tol::Real = 1e-3,
    merge_faces::Bool = true
)
    T = Float64

    # ------------------------------------------------------------------
    # 1. Collect boundary face polygons and normals from mesh B
    # ------------------------------------------------------------------
    nb_b = number_of_boundary_faces(mesh_b)
    bnd_polys = Vector{Vector{SVector{3, T}}}(undef, nb_b)
    bnd_normals = Vector{SVector{3, T}}(undef, nb_b)
    bnd_centroids = Vector{SVector{3, T}}(undef, nb_b)

    for bf in 1:nb_b
        fnodes = collect(mesh_b.boundary_faces.faces_to_nodes[bf])
        pts = SVector{3, T}[mesh_b.node_points[n] for n in fnodes]
        bnd_polys[bf] = pts
        bnd_normals[bf] = polygon_normal(pts)
        bnd_centroids[bf] = sum(pts) / length(pts)
    end

    # Compute bounding box of mesh B for quick rejection
    b_lo = SVector{3, T}(T(Inf), T(Inf), T(Inf))
    b_hi = SVector{3, T}(T(-Inf), T(-Inf), T(-Inf))
    for pt in mesh_b.node_points
        b_lo = min.(b_lo, pt)
        b_hi = max.(b_hi, pt)
    end

    # ------------------------------------------------------------------
    # 2. Cut A with each boundary face of B, keeping both sides.
    #    This splits cells of A that straddle B's boundary without
    #    removing anything yet.
    # ------------------------------------------------------------------
    nc_a = number_of_cells(mesh_a)
    current_mesh = mesh_a
    cell_map = collect(1:nc_a)

    for bf in 1:nb_b
        poly = bnd_polys[bf]
        normal = bnd_normals[bf]
        c = bnd_centroids[bf]

        plane = PlaneCut(c, normal)
        bpoly = _expand_polygon(poly)

        current_mesh, step_info = cut_mesh(current_mesh, plane;
            extra_out = true,
            min_cut_fraction = min_cut_fraction,
            bounding_polygon = bpoly,
            clip_to_polygon = true,
            partial_cut = :none,
            merge_faces = merge_faces
        )
        cell_map = [cell_map[j] for j in step_info["cell_index"]]
    end

    # ------------------------------------------------------------------
    # 3. Remove all cells whose centroids are inside B
    # ------------------------------------------------------------------
    nc_cut = number_of_cells(current_mesh)
    geo = tpfv_geometry(current_mesh)
    keep_cell = trues(nc_cut)

    for c in 1:nc_cut
        cx = geo.cell_centroids[1, c]
        cy = geo.cell_centroids[2, c]
        cz = geo.cell_centroids[3, c]
        centroid = SVector{3, T}(cx, cy, cz)

        # Quick bounding box test
        if any(centroid .< b_lo) || any(centroid .> b_hi)
            continue
        end

        if _point_inside_mesh(centroid, bnd_polys, bnd_normals, bnd_centroids)
            keep_cell[c] = false
        end
    end

    # If there are cells to remove, build trimmed mesh
    trimmed_a = current_mesh
    trimmed_cell_map = cell_map
    if !all(keep_cell)
        trimmed_a, trim_info = _remove_cells(current_mesh, keep_cell)
        trimmed_cell_map = [cell_map[j] for j in trim_info]
    end

    # ------------------------------------------------------------------
    # 4. Glue trimmed A with B
    # ------------------------------------------------------------------
    result, glue_info = glue_mesh(trimmed_a, mesh_b;
        tol = tol,
        face_tol = face_tol,
        coplanar_tol = coplanar_tol,
        extra_out = true
    )

    if extra_out
        nc_result = number_of_cells(result)
        nc_b = number_of_cells(mesh_b)

        cell_origin = Vector{Symbol}(undef, nc_result)
        cell_index_a = Vector{Int}(undef, nc_result)
        cell_index_b = Vector{Int}(undef, nc_result)

        for c in 1:nc_result
            idx_a = glue_info["cell_index_a"][c]
            idx_b = glue_info["cell_index_b"][c]
            if idx_a > 0
                cell_origin[c] = :mesh_a
                cell_index_a[c] = trimmed_cell_map[idx_a]
                cell_index_b[c] = 0
            else
                cell_origin[c] = :mesh_b
                cell_index_a[c] = 0
                cell_index_b[c] = idx_b
            end
        end

        info = Dict{String, Any}(
            "cell_origin" => cell_origin,
            "cell_index_a" => cell_index_a,
            "cell_index_b" => cell_index_b,
        )
        return (result, info)
    end

    return result
end

# ==========================================================================
#  Helpers
# ==========================================================================

"""
    _point_inside_mesh(pt, polys, normals, centroids)

Test whether a 3D point is inside a closed mesh defined by boundary face
polygons.  Uses ray-casting along the +x direction and counts crossings with
boundary face triangles.
"""
function _point_inside_mesh(
    pt::SVector{3, T},
    polys::Vector{Vector{SVector{3, T}}},
    normals::Vector{SVector{3, T}},
    centroids::Vector{SVector{3, T}}
) where T
    crossings = 0
    ray_dir = SVector{3, T}(1, 0, 0)

    for (i, poly) in enumerate(polys)
        n = normals[i]

        # Ray-plane intersection: t = dot(n, centroid - pt) / dot(n, ray_dir)
        denom = dot(n, ray_dir)
        # Skip faces nearly parallel to the ray (scaled epsilon for robustness)
        if abs(denom) < 100 * eps(T)
            continue
        end

        t = dot(n, centroids[i] - pt) / denom
        if t <= 0
            continue  # Behind the ray origin
        end

        # Intersection point
        hit = pt + t * ray_dir

        # Check if hit point is inside the face polygon using
        # projection to the face plane
        if _point_in_face_polygon(hit, poly, n)
            crossings += 1
        end
    end

    return isodd(crossings)
end

"""
    _point_in_face_polygon(pt, poly, normal)

Test whether a 3D point (assumed to be on the polygon's plane) lies inside
the polygon.  Projects to the polygon's local 2D frame and uses ray-casting.
"""
function _point_in_face_polygon(
    pt::SVector{3, T},
    poly::Vector{SVector{3, T}},
    normal::SVector{3, T}
) where T
    # Build local 2D frame
    ref = abs(normal[1]) < 0.9 ? SVector{3, T}(1, 0, 0) : SVector{3, T}(0, 1, 0)
    u = normalize(cross(normal, ref))
    v = cross(normal, u)

    # Project polygon and point to 2D
    c = sum(poly) / length(poly)
    poly_2d = [SVector{2, T}(dot(p - c, u), dot(p - c, v)) for p in poly]
    pt_2d = SVector{2, T}(dot(pt - c, u), dot(pt - c, v))

    return point_in_polygon_2d(pt_2d, poly_2d)
end

"""
    _remove_cells(mesh, keep) -> (new_mesh, cell_mapping)

Remove cells from a mesh.  `keep[c]` is `true` for cells to retain.
Returns the new mesh and a vector mapping new cell indices to old cell indices.
"""
function _remove_cells(
    mesh::UnstructuredMesh{3},
    keep::BitVector
)
    T = eltype(eltype(mesh.node_points))
    nc_old = number_of_cells(mesh)
    nf_old = number_of_faces(mesh)
    nb_old = number_of_boundary_faces(mesh)

    # Build old→new cell mapping
    old_to_new = zeros(Int, nc_old)
    new_count = 0
    cell_mapping = Int[]  # new cell → old cell
    for c in 1:nc_old
        if keep[c]
            new_count += 1
            old_to_new[c] = new_count
            push!(cell_mapping, c)
        end
    end

    # Rebuild face data
    all_face_nodes = Vector{Int}[]
    all_face_neighbors = Tuple{Int, Int}[]
    all_bnd_nodes = Vector{Int}[]
    all_bnd_cells = Int[]

    cell_int_faces = [Int[] for _ in 1:new_count]
    cell_bnd_faces = [Int[] for _ in 1:new_count]

    # Process interior faces
    for f in 1:nf_old
        l_old, r_old = mesh.faces.neighbors[f]
        l_new = old_to_new[l_old]
        r_new = old_to_new[r_old]

        fnodes = collect(mesh.faces.faces_to_nodes[f])

        if l_new > 0 && r_new > 0
            # Both cells kept → interior face
            push!(all_face_nodes, fnodes)
            push!(all_face_neighbors, (l_new, r_new))
            fi = length(all_face_nodes)
            push!(cell_int_faces[l_new], fi)
            push!(cell_int_faces[r_new], fi)
        elseif l_new > 0
            # Only left kept → boundary face
            push!(all_bnd_nodes, fnodes)
            push!(all_bnd_cells, l_new)
            bi = length(all_bnd_nodes)
            push!(cell_bnd_faces[l_new], bi)
        elseif r_new > 0
            # Only right kept → boundary face (reverse node order for outward normal)
            push!(all_bnd_nodes, reverse(fnodes))
            push!(all_bnd_cells, r_new)
            bi = length(all_bnd_nodes)
            push!(cell_bnd_faces[r_new], bi)
        end
        # else: both removed, skip
    end

    # Process boundary faces
    for bf in 1:nb_old
        c_old = mesh.boundary_faces.neighbors[bf]
        c_new = old_to_new[c_old]
        if c_new > 0
            fnodes = collect(mesh.boundary_faces.faces_to_nodes[bf])
            push!(all_bnd_nodes, fnodes)
            push!(all_bnd_cells, c_new)
            bi = length(all_bnd_nodes)
            push!(cell_bnd_faces[c_new], bi)
        end
    end

    # Flatten into CSR format
    faces_nodes = Int[]
    faces_nodespos = Int[1]
    for fnodes in all_face_nodes
        append!(faces_nodes, fnodes)
        push!(faces_nodespos, faces_nodespos[end] + length(fnodes))
    end

    bnd_faces_nodes = Int[]
    bnd_faces_nodespos = Int[1]
    for fnodes in all_bnd_nodes
        append!(bnd_faces_nodes, fnodes)
        push!(bnd_faces_nodespos, bnd_faces_nodespos[end] + length(fnodes))
    end

    cells_faces = Int[]
    cells_facepos = Int[1]
    for c in 1:new_count
        append!(cells_faces, cell_int_faces[c])
        push!(cells_facepos, cells_facepos[end] + length(cell_int_faces[c]))
    end

    bnd_cells_faces = Int[]
    bnd_cells_facepos = Int[1]
    for c in 1:new_count
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
        mesh.node_points,
        all_face_neighbors,
        all_bnd_cells
    )

    return (new_mesh, cell_mapping)
end
