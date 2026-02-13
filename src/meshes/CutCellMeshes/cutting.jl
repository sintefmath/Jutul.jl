struct CutCellInfo
    pos_face_nodes::Vector{Vector{Int}}
    neg_face_nodes::Vector{Vector{Int}}
    cut_face_nodes::Vector{Int}
end

"""
    cut_mesh(mesh::UnstructuredMesh{3}, plane::PlaneCut; kwargs...)
    cut_mesh(mesh::UnstructuredMesh{3}, surface::PolygonalSurface; kwargs...)

Cut an UnstructuredMesh in 3D by a planar constraint. Cells that are
intersected by the plane are split into two new cells, one on each side
of the plane.

- `mesh`: A 3D `UnstructuredMesh`
- `plane` or `surface`: The cutting constraint
- `min_cut_fraction`: Cells where the cut produces a sub-cell with volume
   less than this fraction of the original are left unsplit (default 0.05).
- `bounding_polygon`: Optional vector of 3D points defining a polygon in the
   cutting plane. Only cells whose centroid projects inside this polygon will
   be cut. If `nothing` (default), all intersected cells are cut.
- `clip_to_polygon`: If `true` and `bounding_polygon` is provided, cells that
   are partially inside the bounding polygon (any node projects inside) will
   also be cut. Default is `false`.
- `extra_out`: If `true`, return a tuple `(mesh, info_dict)` where `info_dict`
   contains:
   - `"cell_index"`: Vector mapping each new cell to its original cell index.
   - `"face_index"`: Vector mapping each new interior face to its original face
     index (0 for newly created cut faces).
   - `"boundary_face_index"`: Vector mapping each new boundary face to its
     original boundary face index.
   - `"new_faces"`: Vector of indices of the interior faces added by cutting.

Returns a new `UnstructuredMesh`, or `(UnstructuredMesh, Dict)` if `extra_out=true`.
"""
function cut_mesh(mesh::UnstructuredMesh{3}, surface::PolygonalSurface; extra_out::Bool = false, kwargs...)
    result = mesh
    if extra_out
        nc = number_of_cells(mesh)
        nf = number_of_faces(mesh)
        nb = number_of_boundary_faces(mesh)
        # Initialize with identity mappings
        cell_idx = collect(1:nc)
        face_idx = collect(1:nf)
        bface_idx = collect(1:nb)

        for (i, poly) in enumerate(surface.polygons)
            n = surface.normals[i]
            c = sum(poly) / length(poly)
            plane = PlaneCut(c, n)
            result, step_info = cut_mesh(result, plane; extra_out = true, kwargs...)
            # Compose mappings: new cell → intermediate cell → original cell
            cell_idx = [cell_idx[j] for j in step_info["cell_index"]]
            face_idx = [j == 0 ? 0 : face_idx[j] for j in step_info["face_index"]]
            bface_idx = [j == 0 ? 0 : bface_idx[j] for j in step_info["boundary_face_index"]]
        end

        # new_faces: all faces with no original face (face_index == 0)
        all_new_faces = findall(==(0), face_idx)

        info = Dict{String, Any}(
            "cell_index" => cell_idx,
            "face_index" => face_idx,
            "boundary_face_index" => bface_idx,
            "new_faces" => all_new_faces
        )
        return (result, info)
    else
        for (i, poly) in enumerate(surface.polygons)
            n = surface.normals[i]
            c = sum(poly) / length(poly)
            plane = PlaneCut(c, n)
            result = cut_mesh(result, plane; extra_out = false, kwargs...)
        end
        return result
    end
end

"""
    cut_mesh(mesh, plane; partial_cut=:none, ...)

When `partial_cut` is `:positive` or `:negative`, only cells on the specified
side of the cutting plane are kept after cutting.  Cut cells lose their
sub-cell on the discarded side and the cut face becomes a new boundary face.
Uncut cells that lie entirely on the discarded side are removed.  When
`partial_cut=:none` (default), both sub-cells are kept (standard behaviour).
"""
function cut_mesh(mesh::UnstructuredMesh{3}, plane::PlaneCut{T};
        min_cut_fraction::Real = 0.05,
        bounding_polygon = nothing,
        clip_to_polygon::Bool = false,
        extra_out::Bool = false,
        partial_cut::Symbol = :none
    ) where T
    @assert partial_cut in (:none, :positive, :negative) "partial_cut must be :none, :positive, or :negative"
    nc = number_of_cells(mesh)
    nn = length(mesh.node_points)

    # Classify all nodes
    node_class = Dict{Int, Int}()
    for n in 1:nn
        node_class[n] = classify_point(plane, mesh.node_points[n])
    end

    # Project bounding polygon to 2D if provided
    bpoly_2d = nothing
    if bounding_polygon !== nothing
        bpoly_2d = project_polygon_to_2d(bounding_polygon, plane)
    end

    # Classify all cells
    cut_cells = Int[]
    is_cut = falses(nc)
    for c in 1:nc
        nodes = cell_nodes(mesh, c)
        has_pos = false
        has_neg = false
        for n in nodes
            cl = node_class[n]
            if cl > 0
                has_pos = true
            elseif cl < 0
                has_neg = true
            end
        end
        if has_pos && has_neg
            # Check bounding polygon constraint
            if bpoly_2d !== nothing
                if clip_to_polygon
                    in_bounds = cell_any_node_in_bounding_polygon(mesh, c, plane, bpoly_2d)
                else
                    in_bounds = cell_centroid_in_bounding_polygon(mesh, c, plane, bpoly_2d)
                end
                if !in_bounds
                    continue
                end
            end
            push!(cut_cells, c)
            is_cut[c] = true
        end
    end

    if isempty(cut_cells) && partial_cut == :none
        if extra_out
            nf = number_of_faces(mesh)
            nb = number_of_boundary_faces(mesh)
            info = Dict{String, Any}(
                "cell_index" => collect(1:nc),
                "face_index" => collect(1:nf),
                "boundary_face_index" => collect(1:nb),
                "new_faces" => Int[]
            )
            return (mesh, info)
        end
        return mesh
    end

    if isempty(cut_cells) && partial_cut != :none
        # No cells are cut.  Check whether any cells would be discarded.
        discard = partial_cut == :positive ? :negative : :positive
        any_discarded = false
        for c in 1:nc
            side = classify_cell(mesh, c, plane)
            if side == discard
                any_discarded = true
                break
            end
        end
        if !any_discarded
            # All cells are on the kept side → return unchanged mesh
            if extra_out
                nf = number_of_faces(mesh)
                nb = number_of_boundary_faces(mesh)
                info = Dict{String, Any}(
                    "cell_index" => collect(1:nc),
                    "face_index" => collect(1:nf),
                    "boundary_face_index" => collect(1:nb),
                    "new_faces" => Int[]
                )
                return (mesh, info)
            end
            return mesh
        end
        # Otherwise fall through to build_cut_mesh which will remove
        # cells on the discarded side.
    end

    # Create mutable node list and edge intersection cache
    new_node_points = copy(mesh.node_points)
    edge_cache = Dict{Tuple{Int, Int}, Int}()

    function get_or_create_intersection(n1::Int, n2::Int)
        key = n1 < n2 ? (n1, n2) : (n2, n1)
        if haskey(edge_cache, key)
            return edge_cache[key]
        end
        pt = edge_plane_intersection(new_node_points[n1], new_node_points[n2], plane)
        push!(new_node_points, pt)
        idx = length(new_node_points)
        edge_cache[key] = idx
        node_class[idx] = 0
        return idx
    end

    function split_face(fnodes::AbstractVector{Int})
        nf = length(fnodes)
        has_pos = false
        has_neg = false
        for i in 1:nf
            c = node_class[fnodes[i]]
            if c > 0
                has_pos = true
            elseif c < 0
                has_neg = true
            end
        end
        if !(has_pos && has_neg)
            return nothing
        end

        pos_nodes = Int[]
        neg_nodes = Int[]
        plane_nodes = Int[]

        for i in 1:nf
            j = mod1(i + 1, nf)
            ni = fnodes[i]
            nj = fnodes[j]
            ci = node_class[ni]
            cj = node_class[nj]

            if ci >= 0
                push!(pos_nodes, ni)
            end
            if ci <= 0
                push!(neg_nodes, ni)
            end
            if ci == 0
                push!(plane_nodes, ni)
            end

            if (ci > 0 && cj < 0) || (ci < 0 && cj > 0)
                new_node = get_or_create_intersection(ni, nj)
                push!(pos_nodes, new_node)
                push!(neg_nodes, new_node)
                push!(plane_nodes, new_node)
            end
        end

        return (pos_nodes, neg_nodes, plane_nodes)
    end

    # Build cut cell info for each cut cell
    cut_cell_infos = Dict{Int, CutCellInfo}()

    for cell in cut_cells
        pos_faces = Vector{Int}[]
        neg_faces = Vector{Int}[]
        all_plane_nodes = Int[]

        for face in mesh.faces.cells_to_faces[cell]
            fnodes = collect(mesh.faces.faces_to_nodes[face])
            result = split_face(fnodes)
            if result !== nothing
                pos_fn, neg_fn, pnodes = result
                if length(pos_fn) >= 3
                    push!(pos_faces, pos_fn)
                end
                if length(neg_fn) >= 3
                    push!(neg_faces, neg_fn)
                end
                append!(all_plane_nodes, pnodes)
            else
                side = dominant_side(fnodes, node_class)
                if side >= 0
                    push!(pos_faces, fnodes)
                end
                if side <= 0
                    push!(neg_faces, fnodes)
                end
            end
        end

        for face in mesh.boundary_faces.cells_to_faces[cell]
            fnodes = collect(mesh.boundary_faces.faces_to_nodes[face])
            result = split_face(fnodes)
            if result !== nothing
                pos_fn, neg_fn, pnodes = result
                if length(pos_fn) >= 3
                    push!(pos_faces, pos_fn)
                end
                if length(neg_fn) >= 3
                    push!(neg_faces, neg_fn)
                end
                append!(all_plane_nodes, pnodes)
            else
                side = dominant_side(fnodes, node_class)
                if side >= 0
                    push!(pos_faces, fnodes)
                end
                if side <= 0
                    push!(neg_faces, fnodes)
                end
            end
        end

        # Build cut face from unique plane nodes
        unique_nodes = unique(all_plane_nodes)
        if length(unique_nodes) >= 3
            cut_pts = [new_node_points[n] for n in unique_nodes]
            ordered_pts = order_polygon_points(cut_pts, plane.normal)
            ordered_nodes = Int[]
            for pt in ordered_pts
                for n in unique_nodes
                    if new_node_points[n] ≈ pt
                        push!(ordered_nodes, n)
                        break
                    end
                end
            end
            cut_cell_infos[cell] = CutCellInfo(pos_faces, neg_faces, ordered_nodes)
        else
            is_cut[cell] = false
        end
    end

    # Check volume fractions
    for cell in cut_cells
        if !is_cut[cell]
            continue
        end
        info = cut_cell_infos[cell]
        if isempty(info.pos_face_nodes) || isempty(info.neg_face_nodes)
            is_cut[cell] = false
            continue
        end

        pos_area = sum(polygon_area([new_node_points[n] for n in f]) for f in info.pos_face_nodes; init=0.0)
        neg_area = sum(polygon_area([new_node_points[n] for n in f]) for f in info.neg_face_nodes; init=0.0)
        total_area = pos_area + neg_area
        if total_area > 0
            frac = min(pos_area, neg_area) / total_area
            if frac < min_cut_fraction
                is_cut[cell] = false
            end
        end
    end

    return build_cut_mesh(mesh, plane, new_node_points, is_cut, cut_cell_infos, node_class, get_or_create_intersection, extra_out, partial_cut)
end

"""
    dominant_side(fnodes, node_class)

Determine which side of the plane a face is on.
"""
function dominant_side(fnodes, node_class)
    has_pos = false
    has_neg = false
    for n in fnodes
        c = get(node_class, n, 0)
        if c > 0
            has_pos = true
        elseif c < 0
            has_neg = true
        end
    end
    if has_pos && !has_neg
        return 1
    elseif has_neg && !has_pos
        return -1
    else
        return 0
    end
end

"""
    build_cut_mesh(mesh, plane, node_points, is_cut, cut_infos, node_class, get_intersection, extra_out, partial_cut)

Build the final mesh after cutting.

When `partial_cut` is `:positive` or `:negative`, only cells on the specified
side are kept.  Cut cells lose the sub-cell on the discarded side and the cut
face becomes a boundary face.  Uncut cells entirely on the discarded side are
removed.
"""
function build_cut_mesh(
    mesh::UnstructuredMesh{3},
    plane::PlaneCut,
    node_points::Vector{SVector{3, T}},
    is_cut::BitVector,
    cut_infos::Dict{Int, CutCellInfo},
    node_class::Dict{Int, Int},
    get_intersection::Function,
    extra_out::Bool,
    partial_cut::Symbol = :none
) where T
    nc_old = number_of_cells(mesh)
    nf_old = number_of_faces(mesh)
    nb_old = number_of_boundary_faces(mesh)

    # ------------------------------------------------------------------
    # For partial_cut, classify uncut cells and decide which to discard
    # ------------------------------------------------------------------
    # cell_side[c]: :positive, :negative, or :cut
    cell_side = Vector{Symbol}(undef, nc_old)
    for c in 1:nc_old
        if is_cut[c]
            cell_side[c] = :cut
        else
            cell_side[c] = classify_cell(mesh, c, plane)
        end
    end

    # keep_cell[c] - whether this old cell produces at least one new cell
    keep_cell = trues(nc_old)
    if partial_cut != :none
        discard = partial_cut == :positive ? :negative : :positive
        for c in 1:nc_old
            if !is_cut[c] && cell_side[c] == discard
                keep_cell[c] = false
            end
        end
    end

    # Compute new cell numbering
    new_cell_count = 0
    old_to_new = Dict{Int, Vector{Int}}()

    # Index tracking for extra_out
    cell_index = Int[]

    for c in 1:nc_old
        if !keep_cell[c]
            old_to_new[c] = Int[]
            continue
        end
        if is_cut[c]
            if partial_cut == :none
                pos_cell = new_cell_count + 1
                neg_cell = new_cell_count + 2
                old_to_new[c] = [pos_cell, neg_cell]
                new_cell_count += 2
                push!(cell_index, c)  # pos sub-cell
                push!(cell_index, c)  # neg sub-cell
            elseif partial_cut == :positive
                new_cell_count += 1
                old_to_new[c] = [new_cell_count, 0]  # [pos, no neg]
                push!(cell_index, c)
            else  # :negative
                new_cell_count += 1
                old_to_new[c] = [0, new_cell_count]  # [no pos, neg]
                push!(cell_index, c)
            end
        else
            new_cell_count += 1
            old_to_new[c] = [new_cell_count]
            push!(cell_index, c)
        end
    end

    # New face data
    all_face_nodes = Vector{Int}[]
    all_face_neighbors = Tuple{Int, Int}[]
    all_bnd_nodes = Vector{Int}[]
    all_bnd_cells = Int[]

    cell_int_faces = [Int[] for _ in 1:new_cell_count]
    cell_bnd_faces = [Int[] for _ in 1:new_cell_count]

    # Index tracking for interior faces
    face_index = Int[]        # maps new face → old face (0 for new cut faces)
    new_faces_list = Int[]    # indices of newly created cut faces

    # Index tracking for boundary faces
    bnd_face_index = Int[]    # maps new boundary face → old boundary face

    function add_interior_face!(nodes::Vector{Int}, left::Int, right::Int; old_face::Int = 0)
        push!(all_face_nodes, nodes)
        push!(all_face_neighbors, (left, right))
        fi = length(all_face_nodes)
        push!(cell_int_faces[left], fi)
        push!(cell_int_faces[right], fi)
        push!(face_index, old_face)
        return fi
    end

    function add_boundary_face!(nodes::Vector{Int}, cell::Int; old_bf::Int = 0)
        push!(all_bnd_nodes, nodes)
        push!(all_bnd_cells, cell)
        bi = length(all_bnd_nodes)
        push!(cell_bnd_faces[cell], bi)
        push!(bnd_face_index, old_bf)
        return bi
    end

    function split_face_cached(fnodes::Vector{Int})
        nf = length(fnodes)
        pos_nodes = Int[]
        neg_nodes = Int[]

        for i in 1:nf
            j = mod1(i + 1, nf)
            ni = fnodes[i]
            nj = fnodes[j]
            ci = get(node_class, ni, 0)
            cj = get(node_class, nj, 0)

            if ci >= 0
                push!(pos_nodes, ni)
            end
            if ci <= 0
                push!(neg_nodes, ni)
            end

            if (ci > 0 && cj < 0) || (ci < 0 && cj > 0)
                new_node = get_intersection(ni, nj)
                push!(pos_nodes, new_node)
                push!(neg_nodes, new_node)
            end
        end

        return (pos_nodes, neg_nodes)
    end

    function face_needs_split(fnodes::Vector{Int})
        has_pos = false
        has_neg = false
        for n in fnodes
            c = get(node_class, n, 0)
            if c > 0
                has_pos = true
            elseif c < 0
                has_neg = true
            end
        end
        return has_pos && has_neg
    end

    # Helper: get the new cell index for a given old cell and side
    # For cut cells in partial_cut mode, old_to_new has [pos, neg] where one is 0
    # For cut cells in normal mode, old_to_new has [pos, neg]
    # For uncut cells, old_to_new has [cell]
    function get_new_cell(old_cell::Int, side::Symbol)
        mapping = old_to_new[old_cell]
        if length(mapping) == 0
            return 0  # cell was removed
        elseif length(mapping) == 1
            return mapping[1]  # uncut cell - single new cell
        else
            # Cut cell: mapping = [pos, neg]
            return side == :positive ? mapping[1] : mapping[2]
        end
    end

    # Process interior faces
    for face in 1:nf_old
        l_old, r_old = mesh.faces.neighbors[face]
        fnodes = collect(mesh.faces.faces_to_nodes[face])
        l_cut = is_cut[l_old]
        r_cut = is_cut[r_old]

        if !keep_cell[l_old] && !keep_cell[r_old]
            continue  # both cells removed
        end

        if !l_cut && !r_cut
            l_new = get_new_cell(l_old, cell_side[l_old])
            r_new = get_new_cell(r_old, cell_side[r_old])
            if l_new == 0 && r_new == 0
                continue
            elseif l_new == 0
                # Left cell removed → face becomes boundary on right cell
                add_boundary_face!(fnodes, r_new; old_bf = 0)
            elseif r_new == 0
                # Right cell removed → face becomes boundary on left cell
                add_boundary_face!(fnodes, l_new; old_bf = 0)
            else
                add_interior_face!(fnodes, l_new, r_new; old_face = face)
            end
        elseif l_cut && !r_cut
            r_new = get_new_cell(r_old, cell_side[r_old])
            l_pos = get_new_cell(l_old, :positive)
            l_neg = get_new_cell(l_old, :negative)

            if face_needs_split(fnodes)
                pos_fn, neg_fn = split_face_cached(fnodes)
                if length(pos_fn) >= 3
                    _add_face_or_bnd!(pos_fn, l_pos, r_new, face,
                        add_interior_face!, add_boundary_face!)
                end
                if length(neg_fn) >= 3
                    _add_face_or_bnd!(neg_fn, l_neg, r_new, face,
                        add_interior_face!, add_boundary_face!)
                end
            else
                side = dominant_side(fnodes, node_class)
                if side >= 0
                    _add_face_or_bnd!(fnodes, l_pos, r_new, face,
                        add_interior_face!, add_boundary_face!)
                else
                    _add_face_or_bnd!(fnodes, l_neg, r_new, face,
                        add_interior_face!, add_boundary_face!)
                end
            end
        elseif !l_cut && r_cut
            l_new = get_new_cell(l_old, cell_side[l_old])
            r_pos = get_new_cell(r_old, :positive)
            r_neg = get_new_cell(r_old, :negative)

            if face_needs_split(fnodes)
                pos_fn, neg_fn = split_face_cached(fnodes)
                if length(pos_fn) >= 3
                    _add_face_or_bnd!(pos_fn, l_new, r_pos, face,
                        add_interior_face!, add_boundary_face!)
                end
                if length(neg_fn) >= 3
                    _add_face_or_bnd!(neg_fn, l_new, r_neg, face,
                        add_interior_face!, add_boundary_face!)
                end
            else
                side = dominant_side(fnodes, node_class)
                if side >= 0
                    _add_face_or_bnd!(fnodes, l_new, r_pos, face,
                        add_interior_face!, add_boundary_face!)
                else
                    _add_face_or_bnd!(fnodes, l_new, r_neg, face,
                        add_interior_face!, add_boundary_face!)
                end
            end
        else
            # Both sides cut
            l_pos = get_new_cell(l_old, :positive)
            l_neg = get_new_cell(l_old, :negative)
            r_pos = get_new_cell(r_old, :positive)
            r_neg = get_new_cell(r_old, :negative)

            if face_needs_split(fnodes)
                pos_fn, neg_fn = split_face_cached(fnodes)
                if length(pos_fn) >= 3
                    _add_face_or_bnd!(pos_fn, l_pos, r_pos, face,
                        add_interior_face!, add_boundary_face!)
                end
                if length(neg_fn) >= 3
                    _add_face_or_bnd!(neg_fn, l_neg, r_neg, face,
                        add_interior_face!, add_boundary_face!)
                end
            else
                side = dominant_side(fnodes, node_class)
                if side >= 0
                    _add_face_or_bnd!(fnodes, l_pos, r_pos, face,
                        add_interior_face!, add_boundary_face!)
                elseif side < 0
                    _add_face_or_bnd!(fnodes, l_neg, r_neg, face,
                        add_interior_face!, add_boundary_face!)
                else
                    _add_face_or_bnd!(fnodes, l_pos, r_pos, face,
                        add_interior_face!, add_boundary_face!)
                end
            end
        end
    end

    # Process boundary faces
    for bf in 1:nb_old
        cell_old = mesh.boundary_faces.neighbors[bf]
        fnodes = collect(mesh.boundary_faces.faces_to_nodes[bf])

        if !keep_cell[cell_old]
            continue  # cell removed
        end

        if !is_cut[cell_old]
            cell_new = get_new_cell(cell_old, cell_side[cell_old])
            if cell_new != 0
                add_boundary_face!(fnodes, cell_new; old_bf = bf)
            end
        else
            pos_cell = get_new_cell(cell_old, :positive)
            neg_cell = get_new_cell(cell_old, :negative)

            if face_needs_split(fnodes)
                pos_fn, neg_fn = split_face_cached(fnodes)
                if length(pos_fn) >= 3 && pos_cell != 0
                    add_boundary_face!(pos_fn, pos_cell; old_bf = bf)
                end
                if length(neg_fn) >= 3 && neg_cell != 0
                    add_boundary_face!(neg_fn, neg_cell; old_bf = bf)
                end
            else
                side = dominant_side(fnodes, node_class)
                if side >= 0 && pos_cell != 0
                    add_boundary_face!(fnodes, pos_cell; old_bf = bf)
                elseif side < 0 && neg_cell != 0
                    add_boundary_face!(fnodes, neg_cell; old_bf = bf)
                end
            end
        end
    end

    # Add cut faces
    for c in 1:nc_old
        if !is_cut[c]
            continue
        end
        info = cut_infos[c]
        if length(info.cut_face_nodes) >= 3
            pos_cell = get_new_cell(c, :positive)
            neg_cell = get_new_cell(c, :negative)
            if partial_cut == :none
                # Both sub-cells exist: cut face is interior between them
                fi = add_interior_face!(info.cut_face_nodes, neg_cell, pos_cell; old_face = 0)
                push!(new_faces_list, fi)
            elseif partial_cut == :negative
                # Keeping neg sub-cell: outward normal should point neg→pos
                # (the original ordering already gives that)
                if neg_cell != 0
                    bi = add_boundary_face!(info.cut_face_nodes, neg_cell; old_bf = 0)
                    push!(new_faces_list, bi)
                end
            else
                # Keeping pos sub-cell: outward normal should point pos→neg,
                # i.e. the reverse of the original neg→pos ordering
                if pos_cell != 0
                    bi = add_boundary_face!(reverse(info.cut_face_nodes), pos_cell; old_bf = 0)
                    push!(new_faces_list, bi)
                end
            end
        end
    end

    # Build the mesh arrays
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
        node_points,
        all_face_neighbors,
        all_bnd_cells
    )

    if extra_out
        info_dict = Dict{String, Any}(
            "cell_index" => cell_index,
            "face_index" => face_index,
            "boundary_face_index" => bnd_face_index,
            "new_faces" => new_faces_list
        )
        return (new_mesh, info_dict)
    end
    return new_mesh
end

"""
    _add_face_or_bnd!(nodes, left, right, old_face, add_int!, add_bnd!)

Helper: if both `left` and `right` are valid (non-zero) cell indices, add an
interior face; if only one is valid, add a boundary face on that cell; if
neither is valid, skip.
"""
function _add_face_or_bnd!(nodes, left, right, old_face, add_int!, add_bnd!)
    if left != 0 && right != 0
        add_int!(nodes, left, right; old_face = old_face)
    elseif left != 0
        add_bnd!(nodes, left; old_bf = 0)
    elseif right != 0
        add_bnd!(nodes, right; old_bf = 0)
    end
end
