"""
    refine_mesh_radial(m::UnstructuredMesh{2}, cells; n_sectors = 4, extra_out = false)

Refine selected cells of a 2D `UnstructuredMesh` by splitting each selected cell
into radial (pie-slice) sectors around its centroid. Each original edge of the
cell becomes one triangular sub-cell with vertices at the centroid and the
edge endpoints.

If `n_sectors` is larger than the number of edges, edges are split at their
midpoints to create additional sectors.

# Arguments
- `m`: The 2D mesh to refine.
- `cells`: Collection of cell indices to refine.
- `n_sectors`: Minimum number of radial sectors per cell (default 4).
  If the cell already has at least `n_sectors` edges, one sector per edge
  is created (no edge splitting).
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

    # Determine which cells need edge splitting (n_sectors > n_edges)
    cell_needs_split = Dict{Int, Bool}()
    for c in cells_to_refine
        edges = get_ordered_edges_2d(m, c)
        cell_needs_split[c] = n_sectors > length(edges)
    end

    # Precompute face midpoints only for cells that need splitting
    face_midnode = Dict{Int, Int}()
    bnd_midnode = Dict{Int, Int}()

    for face in 1:nf
        l, r = m.faces.neighbors[face]
        l_needs = (l in cells_to_refine) && cell_needs_split[l]
        r_needs = (r in cells_to_refine) && cell_needs_split[r]
        if l_needs || r_needs
            nodes = m.faces.faces_to_nodes[face]
            @assert length(nodes) == 2
            mid = (m.node_points[nodes[1]] + m.node_points[nodes[2]]) / 2
            push!(new_node_points, PT(mid))
            face_midnode[face] = length(new_node_points)
        end
    end
    for bf in 1:nb
        c = m.boundary_faces.neighbors[bf]
        if (c in cells_to_refine) && cell_needs_split[c]
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
            needs_split = cell_needs_split[c]

            if needs_split
                actual_sectors = 2 * n_edges
            else
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

            if needs_split
                # Build expanded node list: n1, mid, n1, mid, ...
                expanded_nodes = Int[]
                for (n1, n2, _, face_idx, is_bnd) in edges
                    push!(expanded_nodes, n1)
                    if is_bnd
                        push!(expanded_nodes, bnd_midnode[face_idx])
                    else
                        push!(expanded_nodes, face_midnode[face_idx])
                    end
                end

                # Create radial faces between adjacent sectors
                for i in 1:actual_sectors
                    i_next = mod1(i + 1, actual_sectors)
                    node_between = expanded_nodes[i_next]
                    sc_left = sub_cell_ids[i]
                    sc_right = sub_cell_ids[i_next]
                    fi = add_interior_face!(node_between, centroid_node, sc_left, sc_right)
                    push!(cells_to_faces_list[sc_left], fi)
                    push!(cells_to_faces_list[sc_right], fi)
                end
            else
                # No splitting: one sector per edge
                corner_nodes = [e[1] for e in edges]

                for i in 1:n_edges
                    i_next = mod1(i + 1, n_edges)
                    node_between = corner_nodes[i_next]
                    sc_left = sub_cell_ids[i]
                    sc_right = sub_cell_ids[i_next]
                    fi = add_interior_face!(node_between, centroid_node, sc_left, sc_right)
                    push!(cells_to_faces_list[sc_left], fi)
                    push!(cells_to_faces_list[sc_right], fi)
                end
            end
        else
            new_cell_count += 1
            old_cell_to_new[c] = [new_cell_count]
            push!(cells_to_faces_list, Int[])
            push!(cells_to_bnd_list, Int[])
            push!(cell_map, c)
        end
    end

    # Helper: find which sub-cell of a split-refined cell contains a given node
    function find_subcell_for_node_split(cell, node)
        edges = get_ordered_edges_2d(m, cell)
        sub_ids = old_cell_to_new[cell]
        idx = 0
        for (n1, n2, _, face_idx, is_bnd) in edges
            idx += 1
            if n1 == node
                return sub_ids[idx]
            end
            idx += 1
            mid = is_bnd ? bnd_midnode[face_idx] : face_midnode[face_idx]
            if mid == node
                return sub_ids[idx]
            end
        end
        error("Node $node not found in split-refined cell $cell")
    end

    # Helper: find which sub-cell owns a full edge (for non-split cells)
    function find_subcell_for_edge(cell, face_idx, is_boundary)
        edges = get_ordered_edges_2d(m, cell)
        sub_ids = old_cell_to_new[cell]
        for (i, (_, _, _, fi, isbnd)) in enumerate(edges)
            if fi == face_idx && isbnd == is_boundary
                return sub_ids[i]
            end
        end
        error("Edge (face=$face_idx, bnd=$is_boundary) not found in cell $cell")
    end

    # Second pass: handle original interior faces
    for face in 1:nf
        l, r = m.faces.neighbors[face]
        nodes = m.faces.faces_to_nodes[face]
        n1, n2 = nodes[1], nodes[2]

        l_refined = l in cells_to_refine
        r_refined = r in cells_to_refine

        if !l_refined && !r_refined
            new_l = old_cell_to_new[l][1]
            new_r = old_cell_to_new[r][1]
            fi = add_interior_face!(n1, n2, new_l, new_r)
            push!(cells_to_faces_list[new_l], fi)
            push!(cells_to_faces_list[new_r], fi)
        else
            l_split = l_refined && cell_needs_split[l]
            r_split = r_refined && cell_needs_split[r]

            if l_split || r_split
                # At least one side needs edge splitting
                mid = face_midnode[face]

                function get_sc_half(cell, node, refined, split)
                    if !refined
                        return old_cell_to_new[cell][1]
                    elseif split
                        return find_subcell_for_node_split(cell, node)
                    else
                        # Non-split side: the full edge belongs to one sub-cell
                        return find_subcell_for_edge(cell, face, false)
                    end
                end

                sc_l1 = get_sc_half(l, n1, l_refined, l_split)
                sc_r1 = get_sc_half(r, n1, r_refined, r_split)
                fi1 = add_interior_face!(n1, mid, sc_l1, sc_r1)
                push!(cells_to_faces_list[sc_l1], fi1)
                push!(cells_to_faces_list[sc_r1], fi1)

                sc_l2 = get_sc_half(l, mid, l_refined, l_split)
                sc_r2 = get_sc_half(r, mid, r_refined, r_split)
                fi2 = add_interior_face!(mid, n2, sc_l2, sc_r2)
                push!(cells_to_faces_list[sc_l2], fi2)
                push!(cells_to_faces_list[sc_r2], fi2)
            else
                # No edge splitting on either side
                if l_refined
                    new_l = find_subcell_for_edge(l, face, false)
                else
                    new_l = old_cell_to_new[l][1]
                end
                if r_refined
                    new_r = find_subcell_for_edge(r, face, false)
                else
                    new_r = old_cell_to_new[r][1]
                end
                fi = add_interior_face!(n1, n2, new_l, new_r)
                push!(cells_to_faces_list[new_l], fi)
                push!(cells_to_faces_list[new_r], fi)
            end
        end
    end

    # Third pass: handle boundary faces
    for bf in 1:nb
        c = m.boundary_faces.neighbors[bf]
        nodes = m.boundary_faces.faces_to_nodes[bf]
        n1, n2 = nodes[1], nodes[2]

        if c in cells_to_refine
            needs_split = cell_needs_split[c]

            if needs_split
                mid = bnd_midnode[bf]
                sc1 = find_subcell_for_node_split(c, n1)
                sc2 = find_subcell_for_node_split(c, mid)
                bi1 = add_boundary_face!(n1, mid, sc1)
                push!(cells_to_bnd_list[sc1], bi1)
                bi2 = add_boundary_face!(mid, n2, sc2)
                push!(cells_to_bnd_list[sc2], bi2)
            else
                sc = find_subcell_for_edge(c, bf, true)
                bi = add_boundary_face!(n1, n2, sc)
                push!(cells_to_bnd_list[sc], bi)
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
