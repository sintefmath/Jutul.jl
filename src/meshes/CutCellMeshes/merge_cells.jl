"""
    merge_split_cells(mesh, info; category_key="layer_indices")

Merge cells that were unnecessarily split during cutting.  Two cells are
merged when they

  1. originated from the same cell in the input mesh (`info["cell_index"]`),
  2. belong to the same category (`info[category_key]`), and
  3. share at least one interior face.

The merge is transitive (union-find): if cells A–B and B–C satisfy the
conditions they are all merged into one cell.

Returns `(merged_mesh, merged_info)` where `merged_info` contains updated
`"cell_index"` and `category_key` vectors.
"""
function merge_split_cells(
    mesh::UnstructuredMesh{3},
    info::Dict{String, Any};
    category_key::String = "layer_indices"
)
    cell_index = info["cell_index"]::Vector{Int}
    categories = info[category_key]::Vector{Int}
    nc = number_of_cells(mesh)
    @assert length(cell_index) == nc
    @assert length(categories) == nc

    # Union-Find
    parent = collect(1:nc)
    function uf_find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function uf_unite(a, b)
        ra, rb = uf_find(a), uf_find(b)
        if ra != rb
            parent[ra] = rb
        end
    end

    # Unite cells that share a face, same original cell, same category
    nf = number_of_faces(mesh)
    for f in 1:nf
        l, r = mesh.faces.neighbors[f]
        if cell_index[l] == cell_index[r] && categories[l] == categories[r]
            uf_unite(l, r)
        end
    end

    # Build groups
    groups = Dict{Int, Vector{Int}}()
    for c in 1:nc
        r = uf_find(c)
        push!(get!(groups, r, Int[]), c)
    end

    return _do_merge_cells(mesh, groups, cell_index, categories, category_key)
end

"""
    merge_small_cells(mesh, original_mesh, info;
                      threshold=0.1, category_key="layer_indices")

Merge cells whose volume is less than `threshold` times their original cell
volume with the neighbouring cell of the same category that shares the
largest face area.

`original_mesh` is the mesh before cutting.  `info` must contain
`"cell_index"` (mapping to original cells) and a category vector under
`category_key`.

Returns `(merged_mesh, merged_info)`.
"""
function merge_small_cells(
    mesh::UnstructuredMesh{3},
    original_mesh::UnstructuredMesh{3},
    info::Dict{String, Any};
    threshold::Real = 0.1,
    category_key::String = "layer_indices"
)
    cell_index = info["cell_index"]::Vector{Int}
    categories = info[category_key]::Vector{Int}
    nc = number_of_cells(mesh)

    # Compute volumes
    geo = tpfv_geometry(mesh)
    geo_orig = tpfv_geometry(original_mesh)

    # Original cell volumes
    orig_vol = geo_orig.volumes

    # Union-Find
    parent = collect(1:nc)
    function uf_find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function uf_unite(a, b)
        ra, rb = uf_find(a), uf_find(b)
        if ra != rb
            parent[ra] = rb
        end
    end

    # Identify small cells and merge with best neighbour
    nf = number_of_faces(mesh)

    # Build face areas
    face_areas = zeros(nf)
    for f in 1:nf
        face_areas[f] = geo.areas[f]
    end

    # Process cells in order of increasing volume so that small cells get
    # absorbed first.
    order = sortperm(geo.volumes)

    for c in order
        rc = uf_find(c)
        # Check if this cell (or its merged group representative) is small
        v = geo.volumes[c]
        ov = orig_vol[cell_index[c]]
        if ov > 0 && v / ov < threshold
            # Find the neighbor with same category and largest shared face
            best_neighbor = 0
            best_area = 0.0
            for f in mesh.faces.cells_to_faces[c]
                l, r = mesh.faces.neighbors[f]
                other = l == c ? r : l
                ro = uf_find(other)
                if ro != rc && categories[other] == categories[c]
                    a = face_areas[f]
                    if a > best_area
                        best_area = a
                        best_neighbor = other
                    end
                end
            end
            if best_neighbor != 0
                uf_unite(c, best_neighbor)
            end
        end
    end

    # Build groups
    groups = Dict{Int, Vector{Int}}()
    for c in 1:nc
        r = uf_find(c)
        push!(get!(groups, r, Int[]), c)
    end

    return _do_merge_cells(mesh, groups, cell_index, categories, category_key)
end

# =========================================================================
#  Shared implementation: build a new mesh from cell groups
# =========================================================================

"""
    _do_merge_cells(mesh, groups, cell_index, categories, category_key)

Build a new mesh by merging cells according to `groups`.  `groups` maps a
representative cell index to the vector of all cells in that group.
Interior faces between cells in the same group are removed; boundary faces
and inter-group interior faces are preserved.
"""
function _do_merge_cells(
    mesh::UnstructuredMesh{3},
    groups::Dict{Int, Vector{Int}},
    cell_index::Vector{Int},
    categories::Vector{Int},
    category_key::String
)
    T = eltype(eltype(mesh.node_points))
    nc_old = number_of_cells(mesh)
    nf_old = number_of_faces(mesh)
    nb_old = number_of_boundary_faces(mesh)

    # Old cell → new cell mapping
    old_to_new = zeros(Int, nc_old)
    new_cell_index = Int[]
    new_categories = Int[]
    new_cell_id = 0
    for (rep, members) in groups
        new_cell_id += 1
        for c in members
            old_to_new[c] = new_cell_id
        end
        # Use representative cell's metadata
        push!(new_cell_index, cell_index[rep])
        push!(new_categories, categories[rep])
    end
    nc_new = new_cell_id

    # Collect faces
    all_face_nodes = Vector{Int}[]
    all_face_neighbors = Tuple{Int, Int}[]
    all_bnd_nodes = Vector{Int}[]
    all_bnd_cells = Int[]

    cell_int_faces = [Int[] for _ in 1:nc_new]
    cell_bnd_faces = [Int[] for _ in 1:nc_new]

    # Interior faces: keep only if the two neighbors map to different new cells.
    for f in 1:nf_old
        l_old, r_old = mesh.faces.neighbors[f]
        l_new = old_to_new[l_old]
        r_new = old_to_new[r_old]
        if l_new != r_new
            fnodes = _dedup_nodes(collect(mesh.faces.faces_to_nodes[f]))
            if length(fnodes) >= 3
                push!(all_face_nodes, fnodes)
                push!(all_face_neighbors, (l_new, r_new))
                fi = length(all_face_nodes)
                push!(cell_int_faces[l_new], fi)
                push!(cell_int_faces[r_new], fi)
            end
        end
    end

    # Boundary faces: keep all, remap cell
    for bf in 1:nb_old
        c_old = mesh.boundary_faces.neighbors[bf]
        c_new = old_to_new[c_old]
        fnodes = _dedup_nodes(collect(mesh.boundary_faces.faces_to_nodes[bf]))
        if length(fnodes) >= 3
            push!(all_bnd_nodes, fnodes)
            push!(all_bnd_cells, c_new)
            bi = length(all_bnd_nodes)
            push!(cell_bnd_faces[c_new], bi)
        end
    end

    # Build mesh arrays
    new_mesh = _rebuild_mesh_from_data(
        mesh.node_points, nc_new,
        all_face_nodes, all_face_neighbors,
        all_bnd_nodes, all_bnd_cells,
        cell_int_faces, cell_bnd_faces
    )

    new_info = Dict{String, Any}(
        "cell_index" => new_cell_index,
        category_key => new_categories
    )
    return (new_mesh, new_info)
end

"""
    _rebuild_mesh_from_data(node_points, nc, face_nodes, face_neighbors,
                            bnd_nodes, bnd_cells, cell_int_faces, cell_bnd_faces)

Low-level helper: assemble an `UnstructuredMesh` from pre-computed arrays.
"""
function _rebuild_mesh_from_data(
    node_points::Vector{SVector{3, T}},
    nc::Int,
    face_nodes_list::Vector{Vector{Int}},
    face_neighbors::Vector{Tuple{Int, Int}},
    bnd_nodes_list::Vector{Vector{Int}},
    bnd_cells::Vector{Int},
    cell_int_faces::Vector{Vector{Int}},
    cell_bnd_faces::Vector{Vector{Int}}
) where T
    faces_nodes = Int[]
    faces_nodespos = Int[1]
    for fnodes in face_nodes_list
        append!(faces_nodes, fnodes)
        push!(faces_nodespos, faces_nodespos[end] + length(fnodes))
    end

    bnd_faces_nodes = Int[]
    bnd_faces_nodespos = Int[1]
    for fnodes in bnd_nodes_list
        append!(bnd_faces_nodes, fnodes)
        push!(bnd_faces_nodespos, bnd_faces_nodespos[end] + length(fnodes))
    end

    cells_faces = Int[]
    cells_facepos = Int[1]
    for c in 1:nc
        append!(cells_faces, cell_int_faces[c])
        push!(cells_facepos, cells_facepos[end] + length(cell_int_faces[c]))
    end

    bnd_cells_faces = Int[]
    bnd_cells_facepos = Int[1]
    for c in 1:nc
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
        face_neighbors,
        bnd_cells
    )
end

"""
    _dedup_nodes(nodes) -> Vector{Int}

Remove consecutive duplicate node indices from a polygon node list
(treating the list as cyclic).
"""
function _dedup_nodes(nodes::Vector{Int})
    n = length(nodes)
    if n <= 1
        return nodes
    end
    result = Int[]
    for i in 1:n
        j = mod1(i + 1, n)
        if nodes[i] != nodes[j]
            push!(result, nodes[i])
        end
    end
    return result
end
