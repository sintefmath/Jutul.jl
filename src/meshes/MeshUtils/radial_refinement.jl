"""
    refine_mesh_radial(m::UnstructuredMesh{2}, cells; n_sectors = 4, n_rings = 1, center_cell = false, extra_out = false)

Refine selected cells of a 2D `UnstructuredMesh` by splitting each selected cell
into radial sectors and concentric rings around its centroid, producing a
cylinder-like sub-mesh pattern.

# Arguments
- `m`: The 2D mesh to refine.
- `cells`: Collection of cell indices to refine.
- `n_sectors`: Minimum number of radial sectors per cell (default 4).
  If the cell already has at least `n_sectors` edges, one sector per edge
  is created (no edge splitting).
- `n_rings`: Number of concentric rings (default 1). With `n_rings = 1` each
  sector is a single triangle from the cell edge to the centroid. With
  `n_rings > 1`, intermediate rings create quad cells between consecutive
  radial levels and the innermost ring contains triangles to the centroid
  (or a single center cell if `center_cell = true`).
- `center_cell`: If `true`, the innermost region becomes a single polygonal
  cell instead of individual triangular sectors (default `false`).
- `extra_out`: If `true`, return `(mesh = ..., cell_map = ...)`.

# Returns
- The refined `UnstructuredMesh`.
- If `extra_out = true`, a named tuple `(mesh = ..., cell_map = ...)` where
  `cell_map` maps each new cell to its parent in the original mesh.
"""
function refine_mesh_radial(m, cells; kwarg...)
    refine_mesh_radial(UnstructuredMesh(m), cells; kwarg...)
end

function refine_mesh_radial(m::UnstructuredMesh{2}, cells;
        n_sectors = 4, n_rings = 1, center_cell = false, extra_out = false)
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
    cell_centroid_pt = Dict{Int, PT}()
    for c in cells_to_refine
        centroid, _ = compute_centroid_and_measure(m, Cells(), c)
        cpt = PT(centroid)
        cell_centroid_pt[c] = cpt
        push!(new_node_points, cpt)
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

    # For multi-ring: create intermediate ring nodes for each refined cell
    # ring_nodes[c][(sector_idx, ring_level)] = node index
    # ring_level 0 = outer boundary (original nodes), ring_level n_rings = centroid
    # We need intermediate nodes at ring levels 1..n_rings-1
    cell_ring_nodes = Dict{Int, Dict{Tuple{Int,Int}, Int}}()

    for c in cells_to_refine
        edges = get_ordered_edges_2d(m, c)
        needs_split = cell_needs_split[c]
        centroid = cell_centroid_pt[c]

        ring_nodes = Dict{Tuple{Int,Int}, Int}()

        if needs_split
            # Expanded nodes: n1, mid, n1, mid, ...
            outer_nodes = Int[]
            for (n1, n2, _, face_idx, is_bnd) in edges
                push!(outer_nodes, n1)
                if is_bnd
                    push!(outer_nodes, bnd_midnode[face_idx])
                else
                    push!(outer_nodes, face_midnode[face_idx])
                end
            end
        else
            outer_nodes = [e[1] for e in edges]
        end

        nsec = length(outer_nodes)

        # Ring level 0 = outer boundary nodes (already exist)
        for s in 1:nsec
            ring_nodes[(s, 0)] = outer_nodes[s]
        end
        # Ring level n_rings = centroid
        for s in 1:nsec
            ring_nodes[(s, n_rings)] = cell_centroid_node[c]
        end
        # Intermediate ring levels: interpolate between outer node and centroid
        for ring in 1:(n_rings - 1)
            t = ring / n_rings  # fraction toward centroid
            for s in 1:nsec
                outer_pt = new_node_points[outer_nodes[s]]
                intermediate_pt = (1 - t) * outer_pt + t * centroid
                push!(new_node_points, PT(intermediate_pt))
                ring_nodes[(s, ring)] = length(new_node_points)
            end
        end

        cell_ring_nodes[c] = ring_nodes
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

    # Cell layout for each refined cell with S sectors and R rings:
    # sub_cells[(sector, ring)] for ring 1..R-1 (or R if no center_cell)
    # sub_cells[(0, 0)] for center_cell if applicable
    # Ring 1 = outermost (adjacent to original boundary), ring R = innermost

    # Mapping: (sector, ring) -> sub-cell index within this cell's sub-cells
    cell_subcell_map = Dict{Int, Dict{Tuple{Int,Int}, Int}}()

    for c in 1:nc
        if c in cells_to_refine
            edges = get_ordered_edges_2d(m, c)
            needs_split = cell_needs_split[c]
            nsec = needs_split ? 2 * length(edges) : length(edges)

            subcell_map = Dict{Tuple{Int,Int}, Int}()
            sub_cell_ids = Int[]

            if center_cell
                # Rings 1..n_rings-1 have S cells each, plus 1 center cell
                for ring in 1:(n_rings - 1)
                    for s in 1:nsec
                        new_cell_count += 1
                        push!(sub_cell_ids, new_cell_count)
                        push!(cells_to_faces_list, Int[])
                        push!(cells_to_bnd_list, Int[])
                        push!(cell_map, c)
                        subcell_map[(s, ring)] = new_cell_count
                    end
                end
                # Center cell
                new_cell_count += 1
                push!(sub_cell_ids, new_cell_count)
                push!(cells_to_faces_list, Int[])
                push!(cells_to_bnd_list, Int[])
                push!(cell_map, c)
                subcell_map[(0, 0)] = new_cell_count
            else
                # All rings have S cells each
                for ring in 1:n_rings
                    for s in 1:nsec
                        new_cell_count += 1
                        push!(sub_cell_ids, new_cell_count)
                        push!(cells_to_faces_list, Int[])
                        push!(cells_to_bnd_list, Int[])
                        push!(cell_map, c)
                        subcell_map[(s, ring)] = new_cell_count
                    end
                end
            end

            old_cell_to_new[c] = sub_cell_ids
            cell_subcell_map[c] = subcell_map

            ring_nodes = cell_ring_nodes[c]

            # Create internal faces:
            # 1) Radial faces: between adjacent sectors in the same ring
            # 2) Circumferential faces: between adjacent rings in the same sector

            # Radial faces (spoke lines from outer to inner)
            if center_cell
                # Radial faces exist for rings 1..n_rings-1
                for ring in 1:(n_rings - 1)
                    for s in 1:nsec
                        s_next = mod1(s + 1, nsec)
                        # The spoke between sector s and sector s_next at this ring
                        # goes from ring_nodes[(s_next, ring-1)] to ring_nodes[(s_next, ring)]
                        # But actually spokes separate sectors, so spoke for sector boundary
                        # between s and s+1 uses the node at position s+1
                        n_outer = ring_nodes[(s_next, ring - 1)]
                        n_inner = ring_nodes[(s_next, ring)]
                        left_cell = subcell_map[(s, ring)]
                        right_cell = subcell_map[(s_next, ring)]
                        fi = add_interior_face!(n_outer, n_inner, left_cell, right_cell)
                        push!(cells_to_faces_list[left_cell], fi)
                        push!(cells_to_faces_list[right_cell], fi)
                    end
                end
            else
                # Radial faces for all rings 1..n_rings
                for ring in 1:n_rings
                    for s in 1:nsec
                        s_next = mod1(s + 1, nsec)
                        if ring == n_rings
                            # Innermost ring: spokes go from ring boundary to centroid
                            n_outer = ring_nodes[(s_next, ring - 1)]
                            n_inner = cell_centroid_node[c]
                        else
                            n_outer = ring_nodes[(s_next, ring - 1)]
                            n_inner = ring_nodes[(s_next, ring)]
                        end
                        left_cell = subcell_map[(s, ring)]
                        right_cell = subcell_map[(s_next, ring)]
                        fi = add_interior_face!(n_outer, n_inner, left_cell, right_cell)
                        push!(cells_to_faces_list[left_cell], fi)
                        push!(cells_to_faces_list[right_cell], fi)
                    end
                end
            end

            # Circumferential (ring) faces: between ring r and ring r+1 in same sector
            # Node order is reversed (n2, n1) so the face normal points inward
            # (from outer cell toward inner cell / centroid).
            if center_cell
                # Faces between ring r and r+1 for rings 1..n_rings-2
                for ring in 1:(n_rings - 2)
                    for s in 1:nsec
                        s_next = mod1(s + 1, nsec)
                        n1 = ring_nodes[(s, ring)]
                        n2 = ring_nodes[(s_next, ring)]
                        outer_cell = subcell_map[(s, ring)]
                        inner_cell = subcell_map[(s, ring + 1)]
                        fi = add_interior_face!(n2, n1, outer_cell, inner_cell)
                        push!(cells_to_faces_list[outer_cell], fi)
                        push!(cells_to_faces_list[inner_cell], fi)
                    end
                end
                # Faces between innermost ring (n_rings-1) and center cell
                center_id = subcell_map[(0, 0)]
                for s in 1:nsec
                    s_next = mod1(s + 1, nsec)
                    n1 = ring_nodes[(s, n_rings - 1)]
                    n2 = ring_nodes[(s_next, n_rings - 1)]
                    outer_cell = subcell_map[(s, n_rings - 1)]
                    fi = add_interior_face!(n2, n1, outer_cell, center_id)
                    push!(cells_to_faces_list[outer_cell], fi)
                    push!(cells_to_faces_list[center_id], fi)
                end
            else
                # Faces between ring r and r+1 for rings 1..n_rings-1
                for ring in 1:(n_rings - 1)
                    for s in 1:nsec
                        s_next = mod1(s + 1, nsec)
                        n1 = ring_nodes[(s, ring)]
                        n2 = ring_nodes[(s_next, ring)]
                        outer_cell = subcell_map[(s, ring)]
                        inner_cell = subcell_map[(s, ring + 1)]
                        fi = add_interior_face!(n2, n1, outer_cell, inner_cell)
                        push!(cells_to_faces_list[outer_cell], fi)
                        push!(cells_to_faces_list[inner_cell], fi)
                    end
                end
            end
        else
            new_cell_count += 1
            old_cell_to_new[c] = [new_cell_count]
            push!(cells_to_faces_list, Int[])
            push!(cells_to_bnd_list, Int[])
            push!(cell_map, c)
            cell_subcell_map[c] = Dict{Tuple{Int,Int}, Int}()
        end
    end

    # Helper: find outer subcell for a given node on the boundary of a refined cell
    function find_outer_subcell_for_node(cell, node)
        ring_nodes = cell_ring_nodes[cell]
        subcell_map = cell_subcell_map[cell]
        edges = get_ordered_edges_2d(m, cell)
        needs_split = cell_needs_split[cell]
        nsec = needs_split ? 2 * length(edges) : length(edges)

        for s in 1:nsec
            if ring_nodes[(s, 0)] == node
                return subcell_map[(s, 1)]
            end
        end
        error("Node $node not found on outer boundary of cell $cell")
    end

    # Helper: find outer subcell that owns a full edge (for non-split cells)
    function find_outer_subcell_for_edge(cell, face_idx, is_boundary)
        edges = get_ordered_edges_2d(m, cell)
        subcell_map = cell_subcell_map[cell]
        for (i, (_, _, _, fi, isbnd)) in enumerate(edges)
            if fi == face_idx && isbnd == is_boundary
                return subcell_map[(i, 1)]
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
                mid = face_midnode[face]

                function get_sc_half(cell, node, refined, split)
                    if !refined
                        return old_cell_to_new[cell][1]
                    elseif split
                        return find_outer_subcell_for_node(cell, node)
                    else
                        return find_outer_subcell_for_edge(cell, face, false)
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
                if l_refined
                    new_l = find_outer_subcell_for_edge(l, face, false)
                else
                    new_l = old_cell_to_new[l][1]
                end
                if r_refined
                    new_r = find_outer_subcell_for_edge(r, face, false)
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
                sc1 = find_outer_subcell_for_node(c, n1)
                sc2 = find_outer_subcell_for_node(c, mid)
                bi1 = add_boundary_face!(n1, mid, sc1)
                push!(cells_to_bnd_list[sc1], bi1)
                bi2 = add_boundary_face!(mid, n2, sc2)
                push!(cells_to_bnd_list[sc2], bi2)
            else
                sc = find_outer_subcell_for_edge(c, bf, true)
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
