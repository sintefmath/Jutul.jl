"""
    merge_cells(m::UnstructuredMesh, cell_groups; extra_out = false)

Merge groups of cells in an `UnstructuredMesh`. Each group of cells is
combined into a single cell by removing the interior faces between them.
Boundary faces and faces connecting to cells outside the group are preserved.
Nodes that are no longer referenced by any face are removed.

# Arguments
- `m`: The mesh to modify.
- `cell_groups`: A vector of vectors, where each inner vector contains the
  indices of cells to merge into one. E.g. `[[1,2], [5,6,7]]` merges
  cells 1-2 into one cell and cells 5-6-7 into another. Cells not
  mentioned in any group are kept as-is.
- `extra_out`: If `true`, also return a `cell_map` mapping each new cell
  index back to one or more original cell indices.

# Returns
- The merged `UnstructuredMesh`.
- If `extra_out = true`, a named tuple `(mesh = ..., cell_map = ...)` where
  `cell_map` is a `Vector{Vector{Int}}` of length equal to the number of
  cells in the new mesh. Each entry lists the original cell indices that
  were merged into that new cell.
"""
function merge_cells(m, cell_groups; kwarg...)
    merge_cells(UnstructuredMesh(m), cell_groups; kwarg...)
end

function merge_cells(m::UnstructuredMesh, cell_groups::Vector{<:AbstractVector{<:Integer}}; extra_out = false)
    nc = number_of_cells(m)
    nf = number_of_faces(m)
    nb = number_of_boundary_faces(m)

    # Build mapping: original cell -> merged group id (or 0 if ungrouped)
    cell_to_group = zeros(Int, nc)
    for (gid, group) in enumerate(cell_groups)
        for c in group
            @assert 1 <= c <= nc "Cell index $c out of range 1:$nc"
            @assert cell_to_group[c] == 0 "Cell $c appears in multiple groups"
            cell_to_group[c] = gid
        end
    end

    # Assign new cell indices:
    # - Each group gets one new cell
    # - Each ungrouped cell gets one new cell
    ngroups = length(cell_groups)
    # Group g -> new cell index g
    # Ungrouped cells get indices ngroups+1, ngroups+2, ...
    old_to_new = zeros(Int, nc)
    cell_map_out = Vector{Vector{Int}}()

    # Groups first
    for (gid, group) in enumerate(cell_groups)
        for c in group
            old_to_new[c] = gid
        end
        push!(cell_map_out, sort(collect(group)))
    end

    # Ungrouped cells
    next_id = ngroups
    for c in 1:nc
        if cell_to_group[c] == 0
            next_id += 1
            old_to_new[c] = next_id
            push!(cell_map_out, [c])
        end
    end
    new_nc = next_id

    # Classify interior faces
    new_faces_nodes = Int[]
    new_faces_nodespos = Int[1]
    new_neighbors = Tuple{Int, Int}[]

    new_bnd_nodes = Int[]
    new_bnd_nodespos = Int[1]
    new_bnd_cells = Int[]

    cells_to_faces_list = [Int[] for _ in 1:new_nc]
    cells_to_bnd_list = [Int[] for _ in 1:new_nc]

    for face in 1:nf
        l, r = m.faces.neighbors[face]
        new_l = old_to_new[l]
        new_r = old_to_new[r]

        if new_l == new_r
            # Interior face within a merged group - remove it
            continue
        end

        # Keep this face
        nodes = m.faces.faces_to_nodes[face]
        for n in nodes
            push!(new_faces_nodes, n)
        end
        push!(new_faces_nodespos, new_faces_nodespos[end] + length(nodes))
        push!(new_neighbors, (new_l, new_r))
        fi = length(new_neighbors)
        push!(cells_to_faces_list[new_l], fi)
        push!(cells_to_faces_list[new_r], fi)
    end

    # Keep all boundary faces (just remap cell indices)
    for bf in 1:nb
        c = m.boundary_faces.neighbors[bf]
        new_c = old_to_new[c]
        nodes = m.boundary_faces.faces_to_nodes[bf]
        for n in nodes
            push!(new_bnd_nodes, n)
        end
        push!(new_bnd_nodespos, new_bnd_nodespos[end] + length(nodes))
        push!(new_bnd_cells, new_c)
        bi = length(new_bnd_cells)
        push!(cells_to_bnd_list[new_c], bi)
    end

    # Remove unused nodes
    used_nodes = Set{Int}()
    for n in new_faces_nodes
        push!(used_nodes, n)
    end
    for n in new_bnd_nodes
        push!(used_nodes, n)
    end

    sorted_used = sort(collect(used_nodes))
    node_remap = zeros(Int, length(m.node_points))
    for (new_idx, old_idx) in enumerate(sorted_used)
        node_remap[old_idx] = new_idx
    end

    # Remap node indices in faces
    for i in eachindex(new_faces_nodes)
        new_faces_nodes[i] = node_remap[new_faces_nodes[i]]
    end
    for i in eachindex(new_bnd_nodes)
        new_bnd_nodes[i] = node_remap[new_bnd_nodes[i]]
    end

    new_node_points = m.node_points[sorted_used]

    # Build mesh
    cells_faces, cells_facepos = cellmap_to_posmap(cells_to_faces_list, new_nc)
    boundary_cells_faces, boundary_cells_facepos = cellmap_to_posmap(cells_to_bnd_list, new_nc)

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
        return (mesh = new_mesh, cell_map = cell_map_out)
    else
        return new_mesh
    end
end
