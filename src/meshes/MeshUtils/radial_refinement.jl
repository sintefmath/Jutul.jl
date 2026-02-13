"""
    refine_mesh_radial(m::UnstructuredMesh{2}, cells; n_sectors = 4, extra_out = false)

Refine selected cells of a 2D `UnstructuredMesh` by splitting each selected cell
into radial (pie-slice) sectors around its centroid. Each original edge of the
cell becomes one triangular sub-cell.

If `n_sectors` is larger than the number of edges, additional radial cuts are
inserted by subdividing edges.

# Arguments
- `m`: The 2D mesh to refine.
- `cells`: Collection of cell indices to refine.
- `n_sectors`: Minimum number of radial sectors per cell (default 4).
  If the cell already has more edges than `n_sectors`, one sector per edge
  is used.
- `extra_out`: If `true`, return `(mesh = ..., cell_map = ...)`.

# Returns
- The refined `UnstructuredMesh`.
- If `extra_out = true`, a named tuple `(mesh = ..., cell_map = ...)` where
  `cell_map` maps each new cell to its parent in the original mesh.
"""
function refine_mesh_radial(m, cells; kwarg...)
    refine_mesh_radial(UnstructuredMesh(m), cells; kwarg...)
end

function refine_mesh_radial(m::UnstructuredMesh{2}, cells; n_sectors = 4, extra_out = false)
    cells_to_refine = Set{Int}(cells)

    if isempty(cells_to_refine)
        if extra_out
            return (mesh = deepcopy(m), cell_map = collect(1:number_of_cells(m)))
        else
            return deepcopy(m)
        end
    end

    nc = number_of_cells(m)
    nf = number_of_faces(m)
    nb = number_of_boundary_faces(m)
    PT = eltype(m.node_points)

    new_node_points = copy(m.node_points)

    # Precompute cell centroid nodes for refined cells
    cell_centroid_node = Dict{Int, Int}()
    for c in cells_to_refine
        centroid, _ = compute_centroid_and_measure(m, Cells(), c)
        push!(new_node_points, PT(centroid))
        cell_centroid_node[c] = length(new_node_points)
    end

    # For radial refinement with more sectors than edges, we may need
    # face midpoint nodes. Precompute for faces bordering refined cells.
    face_midnode = Dict{Int, Int}()
    bnd_midnode = Dict{Int, Int}()

    for face in 1:nf
        l, r = m.faces.neighbors[face]
        if l in cells_to_refine || r in cells_to_refine
            nodes = m.faces.faces_to_nodes[face]
            @assert length(nodes) == 2
            mid = (m.node_points[nodes[1]] + m.node_points[nodes[2]]) / 2
            push!(new_node_points, PT(mid))
            face_midnode[face] = length(new_node_points)
        end
    end
    for bf in 1:nb
        c = m.boundary_faces.neighbors[bf]
        if c in cells_to_refine
            nodes = m.boundary_faces.faces_to_nodes[bf]
            @assert length(nodes) == 2
            mid = (m.node_points[nodes[1]] + m.node_points[nodes[2]]) / 2
            push!(new_node_points, PT(mid))
            bnd_midnode[bf] = length(new_node_points)
        end
    end

    # Build new mesh structures
    new_faces_nodes = Int[]
    new_faces_nodespos = Int[1]
    new_neighbors = Tuple{Int, Int}[]

    new_bnd_nodes = Int[]
    new_bnd_nodespos = Int[1]
    new_bnd_cells = Int[]

    cells_to_faces_list = Vector{Int}[]
    cells_to_bnd_list = Vector{Int}[]
    cell_map = Int[]

    new_cell_count = 0
    old_cell_to_new = Dict{Int, Vector{Int}}()

    function add_interior_face!(n1, n2, left, right)
        push!(new_faces_nodes, n1, n2)
        push!(new_faces_nodespos, new_faces_nodespos[end] + 2)
        push!(new_neighbors, (left, right))
        return length(new_neighbors)
    end

    function add_boundary_face!(n1, n2, cell)
        push!(new_bnd_nodes, n1, n2)
        push!(new_bnd_nodespos, new_bnd_nodespos[end] + 2)
        push!(new_bnd_cells, cell)
        return length(new_bnd_cells)
    end

    # First pass: create sub-cells
    for c in 1:nc
        if c in cells_to_refine
            edges = get_ordered_edges_2d(m, c)
            n_edges = length(edges)
            centroid_node = cell_centroid_node[c]

            # Determine if we need to split edges to reach n_sectors
            need_split = n_sectors > n_edges

            if need_split
                # Build expanded edge list: split each edge at midpoint
                expanded_edges = Tuple{Int, Int, Int, Int, Bool, Bool}[]  # (n1, n2, sign, face_idx, is_boundary, is_half)
                for (n1, n2, sign, face_idx, is_boundary) in edges
                    if is_boundary
                        mid = bnd_midnode[face_idx]
                    else
                        mid = face_midnode[face_idx]
                    end
                    push!(expanded_edges, (n1, mid, sign, face_idx, is_boundary, true))
                    push!(expanded_edges, (mid, n2, sign, face_idx, is_boundary, true))
                end
                actual_sectors = length(expanded_edges)
            else
                expanded_edges = [(n1, n2, sign, fi, isbnd, false) for (n1, n2, sign, fi, isbnd) in edges]
                actual_sectors = n_edges
            end

            sub_cell_ids = Int[]
            for _ in 1:actual_sectors
                new_cell_count += 1
                push!(sub_cell_ids, new_cell_count)
                push!(cells_to_faces_list, Int[])
                push!(cells_to_bnd_list, Int[])
                push!(cell_map, c)
            end
            old_cell_to_new[c] = sub_cell_ids

            # Create radial faces (centroid to edge endpoints) between adjacent sub-cells
            for i in 1:actual_sectors
                i_next = mod1(i + 1, actual_sectors)
                # The node connecting sector i to sector i_next is expanded_edges[i][2]
                node_between = expanded_edges[i][2]
                sc_left = sub_cell_ids[i]
                sc_right = sub_cell_ids[i_next]
                fi = add_interior_face!(node_between, centroid_node, sc_left, sc_right)
                push!(cells_to_faces_list[sc_left], fi)
                push!(cells_to_faces_list[sc_right], fi)
            end
        else
            new_cell_count += 1
            old_cell_to_new[c] = [new_cell_count]
            push!(cells_to_faces_list, Int[])
            push!(cells_to_bnd_list, Int[])
            push!(cell_map, c)
        end
    end

    # Helper: find which sub-cell of a refined cell owns a given node
    function find_subcell_for_node(cell, node)
        edges = get_ordered_edges_2d(m, cell)
        n_edges = length(edges)
        need_split = n_sectors > n_edges

        if need_split
            # Expanded: 2 sub-cells per edge
            idx = 0
            for (n1, n2, _, face_idx, is_bnd) in edges
                if is_bnd
                    mid = bnd_midnode[face_idx]
                else
                    mid = face_midnode[face_idx]
                end
                idx += 1
                if n1 == node
                    return old_cell_to_new[cell][idx]
                end
                idx += 1
                if mid == node
                    return old_cell_to_new[cell][idx]
                end
            end
        else
            for (i, (n1, _, _, _, _)) in enumerate(edges)
                if n1 == node
                    return old_cell_to_new[cell][i]
                end
            end
        end
        error("Node $node not found in cell $cell")
    end

    # Second pass: handle original interior faces
    for face in 1:nf
        l, r = m.faces.neighbors[face]
        nodes = m.faces.faces_to_nodes[face]
        n1, n2 = nodes[1], nodes[2]

        l_refined = l in cells_to_refine
        r_refined = r in cells_to_refine
        need_split_l = l_refined && n_sectors > length(get_ordered_edges_2d(m, l))
        need_split_r = r_refined && n_sectors > length(get_ordered_edges_2d(m, r))
        need_split_any = need_split_l || need_split_r

        if !l_refined && !r_refined
            new_l = old_cell_to_new[l][1]
            new_r = old_cell_to_new[r][1]
            fi = add_interior_face!(n1, n2, new_l, new_r)
            push!(cells_to_faces_list[new_l], fi)
            push!(cells_to_faces_list[new_r], fi)
        elseif need_split_any
            # Face is split at midpoint
            mid = face_midnode[face]

            function get_subcell(cell, node, is_refined)
                if is_refined
                    return find_subcell_for_node(cell, node)
                else
                    return old_cell_to_new[cell][1]
                end
            end

            sc_l1 = get_subcell(l, n1, l_refined)
            sc_l2 = get_subcell(l, mid, l_refined)
            sc_r1 = get_subcell(r, n1, r_refined)
            sc_r2 = get_subcell(r, mid, r_refined)

            fi1 = add_interior_face!(n1, mid, sc_l1, sc_r1)
            push!(cells_to_faces_list[sc_l1], fi1)
            push!(cells_to_faces_list[sc_r1], fi1)

            fi2 = add_interior_face!(mid, n2, sc_l2, sc_r2)
            push!(cells_to_faces_list[sc_l2], fi2)
            push!(cells_to_faces_list[sc_r2], fi2)
        else
            # At least one side refined, but no edge splitting needed
            function get_sc_nosplit(cell, node, is_refined)
                if is_refined
                    return find_subcell_for_node(cell, node)
                else
                    return old_cell_to_new[cell][1]
                end
            end

            if l_refined && r_refined
                mid = face_midnode[face]
                sc_l1 = find_subcell_for_node(l, n1)
                sc_l2 = find_subcell_for_node(l, n2)
                sc_r1 = find_subcell_for_node(r, n1)
                sc_r2 = find_subcell_for_node(r, n2)
                fi1 = add_interior_face!(n1, mid, sc_l1, sc_r1)
                push!(cells_to_faces_list[sc_l1], fi1)
                push!(cells_to_faces_list[sc_r1], fi1)
                fi2 = add_interior_face!(mid, n2, sc_l2, sc_r2)
                push!(cells_to_faces_list[sc_l2], fi2)
                push!(cells_to_faces_list[sc_r2], fi2)
            elseif l_refined
                mid = face_midnode[face]
                new_r = old_cell_to_new[r][1]
                sc_l1 = find_subcell_for_node(l, n1)
                sc_l2 = find_subcell_for_node(l, n2)
                fi1 = add_interior_face!(n1, mid, sc_l1, new_r)
                push!(cells_to_faces_list[sc_l1], fi1)
                push!(cells_to_faces_list[new_r], fi1)
                fi2 = add_interior_face!(mid, n2, sc_l2, new_r)
                push!(cells_to_faces_list[sc_l2], fi2)
                push!(cells_to_faces_list[new_r], fi2)
            else
                mid = face_midnode[face]
                new_l = old_cell_to_new[l][1]
                sc_r1 = find_subcell_for_node(r, n1)
                sc_r2 = find_subcell_for_node(r, n2)
                fi1 = add_interior_face!(n1, mid, new_l, sc_r1)
                push!(cells_to_faces_list[new_l], fi1)
                push!(cells_to_faces_list[sc_r1], fi1)
                fi2 = add_interior_face!(mid, n2, new_l, sc_r2)
                push!(cells_to_faces_list[new_l], fi2)
                push!(cells_to_faces_list[sc_r2], fi2)
            end
        end
    end

    # Third pass: handle boundary faces
    for bf in 1:nb
        c = m.boundary_faces.neighbors[bf]
        nodes = m.boundary_faces.faces_to_nodes[bf]
        n1, n2 = nodes[1], nodes[2]

        if c in cells_to_refine
            n_edges = length(get_ordered_edges_2d(m, c))
            need_split = n_sectors > n_edges

            if need_split
                mid = bnd_midnode[bf]
                sc1 = find_subcell_for_node(c, n1)
                sc2 = find_subcell_for_node(c, mid)
                bi1 = add_boundary_face!(n1, mid, sc1)
                push!(cells_to_bnd_list[sc1], bi1)
                bi2 = add_boundary_face!(mid, n2, sc2)
                push!(cells_to_bnd_list[sc2], bi2)
            else
                sc1 = find_subcell_for_node(c, n1)
                sc2 = find_subcell_for_node(c, n2)
                mid = bnd_midnode[bf]
                bi1 = add_boundary_face!(n1, mid, sc1)
                push!(cells_to_bnd_list[sc1], bi1)
                bi2 = add_boundary_face!(mid, n2, sc2)
                push!(cells_to_bnd_list[sc2], bi2)
            end
        else
            new_c = old_cell_to_new[c][1]
            bi = add_boundary_face!(n1, n2, new_c)
            push!(cells_to_bnd_list[new_c], bi)
        end
    end

    # Build mesh
    cells_faces, cells_facepos = cellmap_to_posmap(cells_to_faces_list, new_cell_count)
    boundary_cells_faces, boundary_cells_facepos = cellmap_to_posmap(cells_to_bnd_list, new_cell_count)

    new_mesh = UnstructuredMesh(
        cells_faces,
        cells_facepos,
        boundary_cells_faces,
        boundary_cells_facepos,
        new_faces_nodes,
        new_faces_nodespos,
        new_bnd_nodes,
        new_bnd_nodespos,
        new_node_points,
        new_neighbors,
        new_bnd_cells
    )
    if extra_out
        return (mesh = new_mesh, cell_map = cell_map)
    else
        return new_mesh
    end
end
