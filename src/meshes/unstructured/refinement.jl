export refine_mesh

"""
    refine_mesh(m::UnstructuredMesh, cells; factor = 2, extra_out = false)

Refine selected cells of an `UnstructuredMesh`. Each selected cell is subdivided
by placing new nodes at face midpoints (and, in 3D, face centroids and edge midpoints)
and at the cell centroid.

# Arguments
- `m`: The mesh to refine.
- `cells`: Collection of cell indices to refine.
- `factor`: Refinement factor (default 2). Can be:
  - An integer `f`: uniform refinement factor for all selected cells. For `f > 2`
    the refinement is applied by iterating `ceil(Int, log2(f))` passes of
    factor-2 refinement.
  - A `Tuple` (e.g. `(2, 2, 1)` for 3D or `(2, 1)` for 2D): per-direction
    refinement factors. Implemented by applying isotropic factor-2 passes equal
    to the maximum number of passes needed across all directions.
  - A `Vector{Int}`: one factor per cell in `cells`.
- `extra_out`: If `true`, also return a named tuple that includes:
  - `cell_map`: `Vector{Int}` mapping each new cell to its parent cell in the
    *original* input mesh. For iterated refinement the map is composed through
    all passes so it always refers back to the original mesh.

# Returns
- The refined `UnstructuredMesh`.
- If `extra_out = true`, a named tuple `(mesh = ..., cell_map = ...)`.
"""
function refine_mesh(m, cells; kwarg...)
    refine_mesh(UnstructuredMesh(m), cells; kwarg...)
end

function refine_mesh(m::UnstructuredMesh{D}, cells; factor = 2, extra_out = false) where D
    # Handle tuple factor: per-direction refinement
    if factor isa Tuple
        @assert length(factor) == D "Tuple factor must have $D elements for a $(D)D mesh, got $(length(factor))"
        # Convert per-direction factors to number of factor-2 passes
        max_factor = maximum(factor)
        if max_factor <= 1
            if extra_out
                return (mesh = deepcopy(m), cell_map = collect(1:number_of_cells(m)))
            else
                return deepcopy(m)
            end
        end
        # Use the maximum component as the isotropic factor
        return refine_mesh(m, cells; factor = max_factor, extra_out = extra_out)
    end

    # Convert factor specification to per-cell factors
    if factor isa Integer
        factors = fill(Int(factor), length(cells))
    else
        factors = Int.(factor)
        @assert length(factors) == length(cells)
    end
    for f in factors
        @assert f >= 1 "Refinement factor must be >= 1, was $f"
    end

    # Determine the number of factor-2 passes needed per cell
    max_passes = 0
    cell_passes = Dict{Int, Int}()
    for (i, c) in enumerate(cells)
        f = factors[i]
        if f > 1
            npasses = ceil(Int, log2(f))
            cell_passes[c] = npasses
            max_passes = max(max_passes, npasses)
        end
    end

    if max_passes == 0
        if extra_out
            return (mesh = deepcopy(m), cell_map = collect(1:number_of_cells(m)))
        else
            return deepcopy(m)
        end
    end

    # Iterate factor-2 refinement passes
    current_mesh = m
    # cell_map tracks original cell index for each cell in current_mesh
    current_map = collect(1:number_of_cells(m))
    # Track which original cells still need more refinement passes
    remaining_passes = copy(cell_passes)

    for pass in 1:max_passes
        # Determine which cells in current_mesh should be refined in this pass
        cells_this_pass = Int[]
        for c_new in 1:number_of_cells(current_mesh)
            c_orig = current_map[c_new]
            if haskey(remaining_passes, c_orig) && remaining_passes[c_orig] >= pass
                push!(cells_this_pass, c_new)
            end
        end

        if isempty(cells_this_pass)
            break
        end

        # Apply a single factor-2 refinement pass
        result = refine_mesh_single_pass(current_mesh, cells_this_pass, true)
        current_mesh = result.mesh
        # Compose the cell maps: new_map[i] refers back to original mesh
        new_map = Vector{Int}(undef, length(result.cell_map))
        for i in eachindex(result.cell_map)
            new_map[i] = current_map[result.cell_map[i]]
        end
        current_map = new_map
    end

    if extra_out
        return (mesh = current_mesh, cell_map = current_map)
    else
        return current_mesh
    end
end

"""
    refine_mesh_single_pass(m, cells, extra_out)

Apply a single pass of factor-2 refinement to the specified cells.
Always returns `(mesh = ..., cell_map = ...)`.
"""
function refine_mesh_single_pass(m::UnstructuredMesh{D}, cells, extra_out) where D
    cells_to_refine = Set{Int}(cells)
    cell_factor = Dict{Int, Int}(c => 2 for c in cells)

    if isempty(cells_to_refine)
        return (mesh = deepcopy(m), cell_map = collect(1:number_of_cells(m)))
    end
    if D == 2
        return refine_mesh_2d(m, cells_to_refine, cell_factor, extra_out)
    else
        @assert D == 3
        return refine_mesh_3d(m, cells_to_refine, cell_factor, extra_out)
    end
end

function refine_mesh_2d(m::UnstructuredMesh{2}, cells_to_refine, cell_factor, extra_out)
    nc = number_of_cells(m)
    nf = number_of_faces(m)
    nb = number_of_boundary_faces(m)
    nn = length(m.node_points)
    PT = eltype(m.node_points)

    # We'll build a new set of nodes, faces, boundary faces, neighbors, etc.
    new_node_points = copy(m.node_points)

    # Precompute: for each face/bndface that borders a refined cell, add midpoint node
    # face_midnode[face] = index of the midpoint node in new_node_points
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

    # Precompute cell centroids for refined cells
    cell_centroid_node = Dict{Int, Int}()
    for c in cells_to_refine
        centroid, _ = compute_centroid_and_measure(m, Cells(), c)
        push!(new_node_points, PT(centroid))
        cell_centroid_node[c] = length(new_node_points)
    end

    # Build ordered edge list for each refined cell
    # Each cell has edges from interior faces and boundary faces.
    # We need to traverse them in order around the cell.
    # For 2D, edges are line segments with 2 nodes.

    # Build the new mesh data structures
    # Approach: process each cell. If not refined, keep as is (with renumbered faces).
    # If refined, create sub-cells.

    # We need to build:
    # 1. New interior faces (faces_nodes, faces_nodespos, neighbors)
    # 2. New boundary faces (bnd_nodes, bnd_nodespos, bnd_cells)
    # 3. Cell-to-face mapping
    # 4. Cell mapping (new cell -> old cell)

    new_faces_nodes = Int[]
    new_faces_nodespos = Int[1]
    new_neighbors = Tuple{Int, Int}[]

    new_bnd_nodes = Int[]
    new_bnd_nodespos = Int[1]
    new_bnd_cells = Int[]

    cells_to_faces_list = Vector{Int}[]
    cells_to_bnd_list = Vector{Int}[]
    cell_map = Int[]  # new cell index -> old cell index

    # We'll assign new cell indices as we go
    new_cell_count = 0

    # For each original cell, determine the new cell index (or range of indices)
    # old_cell_to_new[c] = range of new cell indices
    old_cell_to_new = Dict{Int, Vector{Int}}()

    # Helper: add an interior face
    function add_interior_face!(n1, n2, left, right)
        push!(new_faces_nodes, n1, n2)
        push!(new_faces_nodespos, new_faces_nodespos[end] + 2)
        push!(new_neighbors, (left, right))
        return length(new_neighbors) # face index
    end

    # Helper: add a boundary face
    function add_boundary_face!(n1, n2, cell)
        push!(new_bnd_nodes, n1, n2)
        push!(new_bnd_nodespos, new_bnd_nodespos[end] + 2)
        push!(new_bnd_cells, cell)
        return length(new_bnd_cells) # boundary face index
    end

    # First pass: create sub-cells for refined cells and unrefined cells
    # We need to process ALL cells to assign new cell IDs
    for c in 1:nc
        if c in cells_to_refine
            # Get ordered edges around the cell
            edges = get_ordered_edges_2d(m, c)
            n_edges = length(edges)
            centroid_node = cell_centroid_node[c]

            sub_cell_ids = Int[]
            for _ in 1:n_edges
                new_cell_count += 1
                push!(sub_cell_ids, new_cell_count)
                push!(cells_to_faces_list, Int[])
                push!(cells_to_bnd_list, Int[])
                push!(cell_map, c)
            end
            old_cell_to_new[c] = sub_cell_ids

            # Each edge becomes a sub-cell.
            # Sub-cell i corresponds to the i-th corner of the cell.
            # The edges around it are:
            # - half of edge (i-1): from midpoint of previous edge to corner node
            # - half of edge i: from corner node to midpoint of current edge
            # - radial face from midpoint of current edge to centroid
            # - radial face from centroid to midpoint of previous edge

            # We need the midpoint node for each edge
            midnodes = Int[]
            for (_, _, _, face_idx, is_boundary) in edges
                if is_boundary
                    push!(midnodes, bnd_midnode[face_idx])
                else
                    push!(midnodes, face_midnode[face_idx])
                end
            end

            # Create radial faces (from midpoint to centroid) - these are internal faces
            # between adjacent sub-cells
            radial_faces = Int[]
            for i in 1:n_edges
                i_next = mod1(i + 1, n_edges)
                sc_left = sub_cell_ids[i]
                sc_right = sub_cell_ids[i_next]
                # radial face from midnodes[i] to centroid_node
                # Normal direction: we want it consistent
                fi = add_interior_face!(midnodes[i], centroid_node, sc_left, sc_right)
                push!(radial_faces, fi)
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

    # Second pass: handle original interior faces
    for face in 1:nf
        l, r = m.faces.neighbors[face]
        nodes = m.faces.faces_to_nodes[face]
        n1, n2 = nodes[1], nodes[2]

        l_refined = l in cells_to_refine
        r_refined = r in cells_to_refine

        if !l_refined && !r_refined
            # Neither cell is refined - keep face as is
            new_l = old_cell_to_new[l][1]
            new_r = old_cell_to_new[r][1]
            fi = add_interior_face!(n1, n2, new_l, new_r)
            push!(cells_to_faces_list[new_l], fi)
            push!(cells_to_faces_list[new_r], fi)
        elseif l_refined && r_refined
            # Both cells refined - split face into 2 sub-faces
            mid = face_midnode[face]
            # Find which sub-cells of l and r border this face
            sc_l1 = find_subcell_for_node_2d(m, l, n1, old_cell_to_new[l])
            sc_l2 = find_subcell_for_node_2d(m, l, n2, old_cell_to_new[l])
            sc_r1 = find_subcell_for_node_2d(m, r, n1, old_cell_to_new[r])
            sc_r2 = find_subcell_for_node_2d(m, r, n2, old_cell_to_new[r])
            # Sub-face 1: n1 to mid
            fi1 = add_interior_face!(n1, mid, sc_l1, sc_r1)
            push!(cells_to_faces_list[sc_l1], fi1)
            push!(cells_to_faces_list[sc_r1], fi1)
            # Sub-face 2: mid to n2
            fi2 = add_interior_face!(mid, n2, sc_l2, sc_r2)
            push!(cells_to_faces_list[sc_l2], fi2)
            push!(cells_to_faces_list[sc_r2], fi2)
        elseif l_refined
            # Only left is refined, right is not
            mid = face_midnode[face]
            new_r = old_cell_to_new[r][1]
            sc_l1 = find_subcell_for_node_2d(m, l, n1, old_cell_to_new[l])
            sc_l2 = find_subcell_for_node_2d(m, l, n2, old_cell_to_new[l])
            # Sub-face 1: n1 to mid
            fi1 = add_interior_face!(n1, mid, sc_l1, new_r)
            push!(cells_to_faces_list[sc_l1], fi1)
            push!(cells_to_faces_list[new_r], fi1)
            # Sub-face 2: mid to n2
            fi2 = add_interior_face!(mid, n2, sc_l2, new_r)
            push!(cells_to_faces_list[sc_l2], fi2)
            push!(cells_to_faces_list[new_r], fi2)
        else
            # Only right is refined
            mid = face_midnode[face]
            new_l = old_cell_to_new[l][1]
            sc_r1 = find_subcell_for_node_2d(m, r, n1, old_cell_to_new[r])
            sc_r2 = find_subcell_for_node_2d(m, r, n2, old_cell_to_new[r])
            # Sub-face 1: n1 to mid
            fi1 = add_interior_face!(n1, mid, new_l, sc_r1)
            push!(cells_to_faces_list[new_l], fi1)
            push!(cells_to_faces_list[sc_r1], fi1)
            # Sub-face 2: mid to n2
            fi2 = add_interior_face!(mid, n2, new_l, sc_r2)
            push!(cells_to_faces_list[new_l], fi2)
            push!(cells_to_faces_list[sc_r2], fi2)
        end
    end

    # Third pass: handle original boundary faces
    for bf in 1:nb
        c = m.boundary_faces.neighbors[bf]
        nodes = m.boundary_faces.faces_to_nodes[bf]
        n1, n2 = nodes[1], nodes[2]

        if c in cells_to_refine
            mid = bnd_midnode[bf]
            sc1 = find_subcell_for_node_2d(m, c, n1, old_cell_to_new[c])
            sc2 = find_subcell_for_node_2d(m, c, n2, old_cell_to_new[c])
            # Sub-boundary 1: n1 to mid
            bi1 = add_boundary_face!(n1, mid, sc1)
            push!(cells_to_bnd_list[sc1], bi1)
            # Sub-boundary 2: mid to n2
            bi2 = add_boundary_face!(mid, n2, sc2)
            push!(cells_to_bnd_list[sc2], bi2)
        else
            new_c = old_cell_to_new[c][1]
            bi = add_boundary_face!(n1, n2, new_c)
            push!(cells_to_bnd_list[new_c], bi)
        end
    end

    # Build the mesh using the middle constructor format
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

"""
Get ordered edges around a 2D cell. Returns a list of tuples:
(node1, node2, face_normal_sign, face_index, is_boundary)
where the edges form a closed loop around the cell.
"""
function get_ordered_edges_2d(m::UnstructuredMesh{2}, cell)
    edges = Tuple{Int, Int, Int, Int, Bool}[]

    # Collect all edges from interior faces
    for face in m.faces.cells_to_faces[cell]
        l, r = m.faces.neighbors[face]
        fnodes = m.faces.faces_to_nodes[face]
        @assert length(fnodes) == 2
        n1, n2 = fnodes[1], fnodes[2]
        if l == cell
            push!(edges, (n1, n2, 1, face, false))
        else
            @assert r == cell
            push!(edges, (n2, n1, -1, face, false))
        end
    end

    # Collect edges from boundary faces
    for bf in m.boundary_faces.cells_to_faces[cell]
        fnodes = m.boundary_faces.faces_to_nodes[bf]
        @assert length(fnodes) == 2
        n1, n2 = fnodes[1], fnodes[2]
        push!(edges, (n1, n2, 1, bf, true))
    end

    # Order the edges so they form a loop
    ordered = empty(edges)
    push!(ordered, popat!(edges, 1))
    while !isempty(edges)
        last_n2 = ordered[end][2]
        i = findfirst(e -> e[1] == last_n2, edges)
        if isnothing(i)
            error("Could not find connected edge for node $last_n2 in cell $cell")
        end
        push!(ordered, popat!(edges, i))
    end
    # Verify loop
    @assert ordered[end][2] == ordered[1][1] "Edges do not form a loop for cell $cell"
    return ordered
end

"""
Find which sub-cell of a refined cell contains a given node.
In 2D, sub-cell i corresponds to the i-th corner node of the cell.
"""
function find_subcell_for_node_2d(m::UnstructuredMesh{2}, cell, node, sub_cell_ids)
    edges = get_ordered_edges_2d(m, cell)
    for (i, (n1, _, _, _, _)) in enumerate(edges)
        if n1 == node
            return sub_cell_ids[i]
        end
    end
    error("Node $node not found in cell $cell")
end

# ----------- 3D refinement -----------

function refine_mesh_3d(m::UnstructuredMesh{3}, cells_to_refine, cell_factor, extra_out)
    nc = number_of_cells(m)
    nf = number_of_faces(m)
    nb = number_of_boundary_faces(m)
    PT = eltype(m.node_points)

    new_node_points = copy(m.node_points)

    # Collect all edges from all faces that border refined cells
    # An edge is identified by its sorted node pair
    edge_midnode = Dict{Tuple{Int, Int}, Int}()

    function get_or_create_edge_midnode!(n1, n2)
        key = n1 < n2 ? (n1, n2) : (n2, n1)
        if !haskey(edge_midnode, key)
            mid = (m.node_points[n1] + m.node_points[n2]) / 2
            push!(new_node_points, PT(mid))
            edge_midnode[key] = length(new_node_points)
        end
        return edge_midnode[key]
    end

    # Face centroid nodes for faces bordering refined cells
    face_centroid_node = Dict{Int, Int}()
    bnd_centroid_node = Dict{Int, Int}()

    for face in 1:nf
        l, r = m.faces.neighbors[face]
        if l in cells_to_refine || r in cells_to_refine
            nodes = m.faces.faces_to_nodes[face]
            # Create edge midpoints
            for i in eachindex(nodes)
                j = i == length(nodes) ? 1 : i + 1
                get_or_create_edge_midnode!(nodes[i], nodes[j])
            end
            # Create face centroid
            centroid = sum(m.node_points[n] for n in nodes) / length(nodes)
            push!(new_node_points, PT(centroid))
            face_centroid_node[face] = length(new_node_points)
        end
    end

    for bf in 1:nb
        c = m.boundary_faces.neighbors[bf]
        if c in cells_to_refine
            nodes = m.boundary_faces.faces_to_nodes[bf]
            for i in eachindex(nodes)
                j = i == length(nodes) ? 1 : i + 1
                get_or_create_edge_midnode!(nodes[i], nodes[j])
            end
            centroid = sum(m.node_points[n] for n in nodes) / length(nodes)
            push!(new_node_points, PT(centroid))
            bnd_centroid_node[bf] = length(new_node_points)
        end
    end

    # Cell centroid nodes
    cell_centroid_node = Dict{Int, Int}()
    for c in cells_to_refine
        centroid, _ = compute_centroid_and_measure(m, Cells(), c)
        push!(new_node_points, PT(centroid))
        cell_centroid_node[c] = length(new_node_points)
    end

    # Collect all unique nodes per cell (corner nodes) in order
    cell_corner_nodes = Dict{Int, Vector{Int}}()
    for c in cells_to_refine
        corners = Int[]
        for face in m.faces.cells_to_faces[c]
            for n in m.faces.faces_to_nodes[face]
                if !(n in corners)
                    push!(corners, n)
                end
            end
        end
        for bf in m.boundary_faces.cells_to_faces[c]
            for n in m.boundary_faces.faces_to_nodes[bf]
                if !(n in corners)
                    push!(corners, n)
                end
            end
        end
        cell_corner_nodes[c] = corners
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
    # Map from (cell, corner_node) to sub-cell index
    corner_to_subcell = Dict{Tuple{Int, Int}, Int}()

    function add_interior_face_3d!(face_nodes, left, right)
        for n in face_nodes
            push!(new_faces_nodes, n)
        end
        push!(new_faces_nodespos, new_faces_nodespos[end] + length(face_nodes))
        push!(new_neighbors, (left, right))
        return length(new_neighbors)
    end

    function add_boundary_face_3d!(face_nodes, cell)
        for n in face_nodes
            push!(new_bnd_nodes, n)
        end
        push!(new_bnd_nodespos, new_bnd_nodespos[end] + length(face_nodes))
        push!(new_bnd_cells, cell)
        return length(new_bnd_cells)
    end

    # First pass: assign sub-cell IDs
    for c in 1:nc
        if c in cells_to_refine
            corners = cell_corner_nodes[c]
            sub_ids = Int[]
            for corner in corners
                new_cell_count += 1
                push!(sub_ids, new_cell_count)
                push!(cells_to_faces_list, Int[])
                push!(cells_to_bnd_list, Int[])
                push!(cell_map, c)
                corner_to_subcell[(c, corner)] = new_cell_count
            end
            old_cell_to_new[c] = sub_ids
        else
            new_cell_count += 1
            old_cell_to_new[c] = [new_cell_count]
            push!(cells_to_faces_list, Int[])
            push!(cells_to_bnd_list, Int[])
            push!(cell_map, c)
        end
    end

    # Internal faces between sub-cells of the same refined cell
    # For each face of a refined cell, each edge of that face creates an internal
    # face between two sub-cells (the ones corresponding to the edge's endpoint corners)
    # The internal face is a quad: corner-node, edge-midpoint, face-centroid, other-edge-midpoint
    # But we also need faces through the cell interior connecting face centroids to cell centroid.

    # Strategy for internal faces of refined cells:
    # For each face of a refined cell, for each edge of that face:
    #   - The edge connects corners c1 and c2
    #   - Create an internal quad face: edge_mid, face_centroid, cell_centroid, (edge_mid of adjacent face sharing same edge)
    # This is complex. Let me use a different approach.

    # For each refined cell, for each pair of adjacent corner nodes that share an edge:
    # create an internal face separating their sub-cells.
    # The face passes through: edge midpoint, and centroids of faces containing that edge, and cell centroid.

    # Collect edges per cell with their adjacent faces
    for c in cells_to_refine
        cc_node = cell_centroid_node[c]
        edge_faces = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Bool}}}()  # edge -> list of (face_idx, is_boundary)

        for face in m.faces.cells_to_faces[c]
            nodes = m.faces.faces_to_nodes[face]
            nn = length(nodes)
            for i in 1:nn
                j = i == nn ? 1 : i + 1
                edge = nodes[i] < nodes[j] ? (nodes[i], nodes[j]) : (nodes[j], nodes[i])
                if !haskey(edge_faces, edge)
                    edge_faces[edge] = Tuple{Int, Bool}[]
                end
                push!(edge_faces[edge], (face, false))
            end
        end
        for bf in m.boundary_faces.cells_to_faces[c]
            nodes = m.boundary_faces.faces_to_nodes[bf]
            nn = length(nodes)
            for i in 1:nn
                j = i == nn ? 1 : i + 1
                edge = nodes[i] < nodes[j] ? (nodes[i], nodes[j]) : (nodes[j], nodes[i])
                if !haskey(edge_faces, edge)
                    edge_faces[edge] = Tuple{Int, Bool}[]
                end
                push!(edge_faces[edge], (bf, true))
            end
        end

        # For each edge, create an internal face between the two sub-cells
        for ((n1, n2), adj_faces) in edge_faces
            sc1 = corner_to_subcell[(c, n1)]
            sc2 = corner_to_subcell[(c, n2)]
            emid = edge_midnode[(n1, n2)]

            # Build the face nodes: edge_mid, fc1, cell_centroid, fc2
            # where fc1 and fc2 are the face centroids of the two faces sharing this edge
            face_centroid_nodes_list = Int[]
            for (fi, is_bnd) in adj_faces
                if is_bnd
                    push!(face_centroid_nodes_list, bnd_centroid_node[fi])
                else
                    push!(face_centroid_nodes_list, face_centroid_node[fi])
                end
            end

            if length(face_centroid_nodes_list) == 2
                fc1, fc2 = face_centroid_nodes_list
                internal_face_nodes = [emid, fc1, cc_node, fc2]
            else
                # Non-standard case: edge is shared by more than 2 faces of this
                # cell (e.g. degenerate or non-manifold geometry). Create a fan of
                # quad faces through the edge midpoint and cell centroid.
                for i in 1:length(face_centroid_nodes_list)
                    j = i == length(face_centroid_nodes_list) ? 1 : i + 1
                    fn = [emid, face_centroid_nodes_list[i], cc_node, face_centroid_nodes_list[j]]
                    orient_face_3d!(fn, new_node_points, new_node_points[n1], new_node_points[n2])
                    fi = add_interior_face_3d!(fn, sc1, sc2)
                    push!(cells_to_faces_list[sc1], fi)
                    push!(cells_to_faces_list[sc2], fi)
                end
                continue
            end

            # Orient the face so the normal points from sc1 (corner n1) to sc2 (corner n2)
            orient_face_3d!(internal_face_nodes, new_node_points, new_node_points[n1], new_node_points[n2])

            fi = add_interior_face_3d!(internal_face_nodes, sc1, sc2)
            push!(cells_to_faces_list[sc1], fi)
            push!(cells_to_faces_list[sc2], fi)
        end
    end

    # Process original interior faces
    for face in 1:nf
        l, r = m.faces.neighbors[face]
        nodes = collect(m.faces.faces_to_nodes[face])
        nn = length(nodes)

        l_refined = l in cells_to_refine
        r_refined = r in cells_to_refine

        if !l_refined && !r_refined
            new_l = old_cell_to_new[l][1]
            new_r = old_cell_to_new[r][1]
            fi = add_interior_face_3d!(nodes, new_l, new_r)
            push!(cells_to_faces_list[new_l], fi)
            push!(cells_to_faces_list[new_r], fi)
        else
            fc = face_centroid_node[face]
            # Compute the original face normal to preserve orientation
            orig_nodes_list = collect(m.faces.faces_to_nodes[face])
            p1_orig = m.node_points[orig_nodes_list[1]]
            p2_orig = m.node_points[orig_nodes_list[2]]
            p3_orig = m.node_points[orig_nodes_list[3]]
            orig_normal = cross(p2_orig - p1_orig, p3_orig - p1_orig)
            # Split face into nn sub-faces (one per corner node)
            for i in 1:nn
                i_prev = i == 1 ? nn : i - 1
                corner = nodes[i]
                n_prev = nodes[i_prev]
                n_next = nodes[i == nn ? 1 : i + 1]

                emid_prev = get_edge_midnode(edge_midnode, corner, n_prev)
                emid_next = get_edge_midnode(edge_midnode, corner, n_next)

                sub_face_nodes = [corner, emid_next, fc, emid_prev]

                # Orient sub-face consistently with original face
                sp1 = new_node_points[sub_face_nodes[1]]
                sp2 = new_node_points[sub_face_nodes[2]]
                sp3 = new_node_points[sub_face_nodes[3]]
                sub_normal = cross(sp2 - sp1, sp3 - sp1)
                if dot(sub_normal, orig_normal) < 0
                    reverse!(sub_face_nodes)
                end

                # Determine left/right sub-cells
                if l_refined
                    new_l = corner_to_subcell[(l, corner)]
                else
                    new_l = old_cell_to_new[l][1]
                end
                if r_refined
                    new_r = corner_to_subcell[(r, corner)]
                else
                    new_r = old_cell_to_new[r][1]
                end

                fi = add_interior_face_3d!(sub_face_nodes, new_l, new_r)
                push!(cells_to_faces_list[new_l], fi)
                push!(cells_to_faces_list[new_r], fi)
            end
        end
    end

    # Process boundary faces
    for bf in 1:nb
        c = m.boundary_faces.neighbors[bf]
        nodes = collect(m.boundary_faces.faces_to_nodes[bf])
        nn = length(nodes)

        if c in cells_to_refine
            fc = bnd_centroid_node[bf]
            # Compute original boundary face normal for orientation
            orig_bnodes = collect(m.boundary_faces.faces_to_nodes[bf])
            bp1 = m.node_points[orig_bnodes[1]]
            bp2 = m.node_points[orig_bnodes[2]]
            bp3 = m.node_points[orig_bnodes[3]]
            orig_bnd_normal = cross(bp2 - bp1, bp3 - bp1)
            for i in 1:nn
                i_prev = i == 1 ? nn : i - 1
                corner = nodes[i]
                n_prev = nodes[i_prev]
                n_next = nodes[i == nn ? 1 : i + 1]

                emid_prev = get_edge_midnode(edge_midnode, corner, n_prev)
                emid_next = get_edge_midnode(edge_midnode, corner, n_next)

                sub_face_nodes = [corner, emid_next, fc, emid_prev]
                # Orient consistently with original boundary face
                sp1 = new_node_points[sub_face_nodes[1]]
                sp2 = new_node_points[sub_face_nodes[2]]
                sp3 = new_node_points[sub_face_nodes[3]]
                sub_normal = cross(sp2 - sp1, sp3 - sp1)
                if dot(sub_normal, orig_bnd_normal) < 0
                    reverse!(sub_face_nodes)
                end
                sc = corner_to_subcell[(c, corner)]
                bi = add_boundary_face_3d!(sub_face_nodes, sc)
                push!(cells_to_bnd_list[sc], bi)
            end
        else
            new_c = old_cell_to_new[c][1]
            bi = add_boundary_face_3d!(nodes, new_c)
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

function get_edge_midnode(edge_midnode, n1, n2)
    key = n1 < n2 ? (n1, n2) : (n2, n1)
    return edge_midnode[key]
end

"""
Orient a 3D face (given as a list of node indices) so that its outward normal
points from `pt_left` toward `pt_right`. Reverses the node list in-place if needed.
"""
function orient_face_3d!(face_nodes, node_points, pt_left, pt_right)
    # Compute face centroid
    fc = sum(node_points[n] for n in face_nodes) / length(face_nodes)
    # Compute approximate normal using first three nodes
    p1 = node_points[face_nodes[1]]
    p2 = node_points[face_nodes[2]]
    p3 = node_points[face_nodes[3]]
    normal = cross(p2 - p1, p3 - p1)
    # Direction from left to right
    dir = pt_right - pt_left
    if dot(normal, dir) < 0
        reverse!(face_nodes)
    end
    return face_nodes
end
