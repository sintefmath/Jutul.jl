struct CutCellInfo
    pos_face_nodes::Vector{Vector{Int}}
    neg_face_nodes::Vector{Vector{Int}}
    cut_face_nodes::Vector{Int}
end

"""
    cut_mesh(mesh::UnstructuredMesh{3}, plane::PlaneCut; min_cut_fraction=0.05)
    cut_mesh(mesh::UnstructuredMesh{3}, surface::PolygonalSurface; min_cut_fraction=0.05)

Cut an UnstructuredMesh in 3D by a planar constraint. Cells that are
intersected by the plane are split into two new cells, one on each side
of the plane.

- `mesh`: A 3D `UnstructuredMesh`
- `plane` or `surface`: The cutting constraint
- `min_cut_fraction`: Cells where the cut produces a sub-cell with volume
   less than this fraction of the original are left unsplit.

Returns a new `UnstructuredMesh`.
"""
function cut_mesh(mesh::UnstructuredMesh{3}, surface::PolygonalSurface; kwarg...)
    result = mesh
    for (i, poly) in enumerate(surface.polygons)
        n = surface.normals[i]
        c = sum(poly) / length(poly)
        plane = PlaneCut(c, n)
        result = cut_mesh(result, plane; kwarg...)
    end
    return result
end

function cut_mesh(mesh::UnstructuredMesh{3}, plane::PlaneCut{T}; min_cut_fraction::Real = 0.05) where T
    nc = number_of_cells(mesh)
    nn = length(mesh.node_points)

    # Classify all nodes
    node_class = Dict{Int, Int}()
    for n in 1:nn
        node_class[n] = classify_point(plane, mesh.node_points[n])
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
            push!(cut_cells, c)
            is_cut[c] = true
        end
    end

    if isempty(cut_cells)
        return mesh
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

    return build_cut_mesh(mesh, plane, new_node_points, is_cut, cut_cell_infos, node_class, get_or_create_intersection)
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
    build_cut_mesh(mesh, plane, node_points, is_cut, cut_infos, node_class, get_intersection)

Build the final mesh after cutting.
"""
function build_cut_mesh(
    mesh::UnstructuredMesh{3},
    plane::PlaneCut,
    node_points::Vector{SVector{3, T}},
    is_cut::BitVector,
    cut_infos::Dict{Int, CutCellInfo},
    node_class::Dict{Int, Int},
    get_intersection::Function
) where T
    nc_old = number_of_cells(mesh)
    nf_old = number_of_faces(mesh)
    nb_old = number_of_boundary_faces(mesh)

    # Compute new cell numbering
    new_cell_count = 0
    old_to_new = Dict{Int, Vector{Int}}()

    for c in 1:nc_old
        if is_cut[c]
            pos_cell = new_cell_count + 1
            neg_cell = new_cell_count + 2
            old_to_new[c] = [pos_cell, neg_cell]
            new_cell_count += 2
        else
            new_cell_count += 1
            old_to_new[c] = [new_cell_count]
        end
    end

    # New face data
    all_face_nodes = Vector{Int}[]
    all_face_neighbors = Tuple{Int, Int}[]
    all_bnd_nodes = Vector{Int}[]
    all_bnd_cells = Int[]

    cell_int_faces = [Int[] for _ in 1:new_cell_count]
    cell_bnd_faces = [Int[] for _ in 1:new_cell_count]

    function add_interior_face!(nodes::Vector{Int}, left::Int, right::Int)
        push!(all_face_nodes, nodes)
        push!(all_face_neighbors, (left, right))
        fi = length(all_face_nodes)
        push!(cell_int_faces[left], fi)
        push!(cell_int_faces[right], fi)
        return fi
    end

    function add_boundary_face!(nodes::Vector{Int}, cell::Int)
        push!(all_bnd_nodes, nodes)
        push!(all_bnd_cells, cell)
        bi = length(all_bnd_nodes)
        push!(cell_bnd_faces[cell], bi)
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

    # Process interior faces
    for face in 1:nf_old
        l_old, r_old = mesh.faces.neighbors[face]
        fnodes = collect(mesh.faces.faces_to_nodes[face])
        l_cut = is_cut[l_old]
        r_cut = is_cut[r_old]

        if !l_cut && !r_cut
            l_new = old_to_new[l_old][1]
            r_new = old_to_new[r_old][1]
            add_interior_face!(fnodes, l_new, r_new)
        elseif l_cut && !r_cut
            r_new = old_to_new[r_old][1]
            l_pos = old_to_new[l_old][1]
            l_neg = old_to_new[l_old][2]

            if face_needs_split(fnodes)
                pos_fn, neg_fn = split_face_cached(fnodes)
                if length(pos_fn) >= 3
                    add_interior_face!(pos_fn, l_pos, r_new)
                end
                if length(neg_fn) >= 3
                    add_interior_face!(neg_fn, l_neg, r_new)
                end
            else
                side = dominant_side(fnodes, node_class)
                if side >= 0
                    add_interior_face!(fnodes, l_pos, r_new)
                else
                    add_interior_face!(fnodes, l_neg, r_new)
                end
            end
        elseif !l_cut && r_cut
            l_new = old_to_new[l_old][1]
            r_pos = old_to_new[r_old][1]
            r_neg = old_to_new[r_old][2]

            if face_needs_split(fnodes)
                pos_fn, neg_fn = split_face_cached(fnodes)
                if length(pos_fn) >= 3
                    add_interior_face!(pos_fn, l_new, r_pos)
                end
                if length(neg_fn) >= 3
                    add_interior_face!(neg_fn, l_new, r_neg)
                end
            else
                side = dominant_side(fnodes, node_class)
                if side >= 0
                    add_interior_face!(fnodes, l_new, r_pos)
                else
                    add_interior_face!(fnodes, l_new, r_neg)
                end
            end
        else
            # Both sides cut
            l_pos = old_to_new[l_old][1]
            l_neg = old_to_new[l_old][2]
            r_pos = old_to_new[r_old][1]
            r_neg = old_to_new[r_old][2]

            if face_needs_split(fnodes)
                pos_fn, neg_fn = split_face_cached(fnodes)
                if length(pos_fn) >= 3
                    add_interior_face!(pos_fn, l_pos, r_pos)
                end
                if length(neg_fn) >= 3
                    add_interior_face!(neg_fn, l_neg, r_neg)
                end
            else
                side = dominant_side(fnodes, node_class)
                if side >= 0
                    add_interior_face!(fnodes, l_pos, r_pos)
                elseif side < 0
                    add_interior_face!(fnodes, l_neg, r_neg)
                else
                    add_interior_face!(fnodes, l_pos, r_pos)
                end
            end
        end
    end

    # Process boundary faces
    for bf in 1:nb_old
        cell_old = mesh.boundary_faces.neighbors[bf]
        fnodes = collect(mesh.boundary_faces.faces_to_nodes[bf])

        if !is_cut[cell_old]
            cell_new = old_to_new[cell_old][1]
            add_boundary_face!(fnodes, cell_new)
        else
            pos_cell = old_to_new[cell_old][1]
            neg_cell = old_to_new[cell_old][2]

            if face_needs_split(fnodes)
                pos_fn, neg_fn = split_face_cached(fnodes)
                if length(pos_fn) >= 3
                    add_boundary_face!(pos_fn, pos_cell)
                end
                if length(neg_fn) >= 3
                    add_boundary_face!(neg_fn, neg_cell)
                end
            else
                side = dominant_side(fnodes, node_class)
                if side >= 0
                    add_boundary_face!(fnodes, pos_cell)
                else
                    add_boundary_face!(fnodes, neg_cell)
                end
            end
        end
    end

    # Add cut faces (internal faces between pos and neg sub-cells)
    for c in 1:nc_old
        if !is_cut[c]
            continue
        end
        info = cut_infos[c]
        if length(info.cut_face_nodes) >= 3
            pos_cell = old_to_new[c][1]
            neg_cell = old_to_new[c][2]
            add_interior_face!(copy(info.cut_face_nodes), neg_cell, pos_cell)
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

    return UnstructuredMesh(
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
end
