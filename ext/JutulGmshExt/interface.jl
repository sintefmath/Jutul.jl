
"""
    Jutul.mesh_from_gmsh("path/to/file.geo", manage_gmsh = true, verbose = false)

Convert Gmsh mesh file to Jutul mesh. If `manage_gmsh` is true, the Gmsh API is
initialized and finalized automatically. Otherwise, the user is responsible for
calling `Gmsh.initialize()` and `Gmsh.finalize()` before and after this
function, respectively.

To use this function, you need to have the Gmsh library installed and loaded by
calling `using Gmsh`. Please note that, unlike Jutul, Gmsh is GPL licensed
software, and you should comply with the license terms when using it in your
projects.

# Keyword arguments
- `argv::Vector{String}`: Command-line arguments to pass to Gmsh during
  initialization. Example: `["-v", "2"]` to set verbosity level to 2.
- `manage_gmsh::Bool`: Whether to initialize and finalize Gmsh automatically.
- `verbose::Bool`: Whether to print messages about the mesh parsing process.
- `reverse_z::Bool`: Whether to reverse the z-coordinates of the mesh nodes.
- `z_is_depth::Bool`: Whether the z-coordinates represent depth (positive
  downwards), passed onto the mesh constructor.
- `remove_duplicate_nodes::Bool`: Whether to remove duplicate nodes in the mesh.
- `preserve_order::Bool`: Whether to preserve the original cell ordering based
  on the Gmsh tags.
"""
function Jutul.mesh_from_gmsh(pth;
        argv = String[],
        manage_gmsh = true,
        verbose = false,
        kwarg...
    )
    if !("v" in argv)
        if verbose
            lvl = 5
        else
            lvl = 0
        end
        push!(argv, "-v", string(lvl))
    end
    if manage_gmsh
        Gmsh.initialize(argv)
    end
    ext = pth |> splitext |> last
    g = missing
    try
        gmsh.open(pth)
        if lowercase(ext) == ".geo"
            gmsh.model.mesh.generate()
        end
        g = Jutul.mesh_from_gmsh(; verbose = verbose, kwarg...)
    finally
        if manage_gmsh
            Gmsh.gmsh.clear()
            Gmsh.finalize()
        end
    end
    if ismissing(g)
        error("Failed to parse mesh")
    end
    return g
end

function Jutul.mesh_from_gmsh(;
        verbose = false,
        reverse_z = false,
        remove_duplicate_nodes = true,
        preserve_order = false,
        kwarg...
    )
    dim = gmsh.model.getDimension()
    dim == 3 || error("Only 3D models are supported")
    if preserve_order
        # Grab initial cell tags to preserve order later.
        tag2cell = get_cell_tags()
    else
        tag2cell = missing
    end
    if remove_duplicate_nodes
        gmsh.model.mesh.removeDuplicateNodes()
    end
    if reverse_z
        # Note: Gmsh API lets us send only the first 3 rows of the 4 by 4 matrix
        # which is sufficient here.
        gmsh.model.mesh.affineTransform(
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, -1.0, 0.0
            ]
        )
        gmsh.model.mesh.generate()
    end
    node_tags, pts, = gmsh.model.mesh.getNodes()
    node_remap = Dict{UInt64, Int}()
    for (i, tag) in enumerate(node_tags)
        tag::UInt64
        node_remap[tag] = i
    end
    remaps = (
        nodes = node_remap,
        faces = Dict{UInt64, Int}(),
        cells = Dict{UInt64, Int}()
    )
    pts = reshape(pts, Int(dim), :)
    pts_s = collect(vec(reinterpret(SVector{3, Float64}, pts)))

    @assert size(pts, 2) == length(node_tags)
    faces_to_nodes = parse_faces(remaps, verbose = verbose)
    face_lookup = generate_face_lookup(faces_to_nodes)

    cells_to_faces = parse_cells(remaps, faces_to_nodes, face_lookup, verbose = verbose)
    neighbors = build_neighbors(cells_to_faces, faces_to_nodes, face_lookup)

    # Make both of these in case we have rogue faces that are not connected to any cell.
    bnd_faces = Int[]
    int_faces = Int[]
    n_bad = 0
    for i in axes(neighbors, 2)
        l, r = neighbors[:, i]
        l_bnd = l == 0
        r_bnd = r == 0

        if l_bnd || r_bnd 
            if l_bnd && r_bnd
                n_bad += 1
            else
                push!(bnd_faces, i)
            end
        else
            push!(int_faces, i)
        end
    end
    bnd_neighbors, bnd_faces_to_nodes, bnd_cells_to_faces = split_boundary(neighbors, faces_to_nodes, cells_to_faces, bnd_faces, boundary = true)
    int_neighbors, int_faces_to_nodes, int_cells_to_faces = split_boundary(neighbors, faces_to_nodes, cells_to_faces, int_faces, boundary = false)

    if preserve_order
        # We have to remap cells here based on the original tags.
        cell_tags = keys(remaps.cells)
        nc = length(int_cells_to_faces)
        if length(cell_tags) == nc
            cell_tags_sorted = sort(collect(cell_tags))
            cell_idx_to_new_idx = zeros(Int, nc)
            new_idx_to_cell_idx = zeros(Int, nc)
            for tag in cell_tags_sorted
                new_cell_idx = tag2cell[tag]
                @assert new_cell_idx > 0 && new_cell_idx <= nc
                current_cell_idx = remaps.cells[tag]
                cell_idx_to_new_idx[current_cell_idx] = new_cell_idx
                new_idx_to_cell_idx[new_cell_idx] = current_cell_idx
            end
            int_cells_to_faces = int_cells_to_faces[new_idx_to_cell_idx]
            bnd_cells_to_faces = bnd_cells_to_faces[new_idx_to_cell_idx]
            bnd_neighbors = cell_idx_to_new_idx[bnd_neighbors]
            int_neighbors = map(lr -> (cell_idx_to_new_idx[lr[1]], cell_idx_to_new_idx[lr[2]]), int_neighbors)
        else
            # Warn about missing tags
            @warn "Number of cell tags ($(length(cell_tags))) does not match number of cells ($nc), cannot preserve cell tags."
        end
    end

    c2f = IndirectionMap(int_cells_to_faces)
    c2b = IndirectionMap(bnd_cells_to_faces)
    f2n = IndirectionMap(int_faces_to_nodes)
    b2n = IndirectionMap(bnd_faces_to_nodes)
    print_message("Mesh parsed successfully:\n    $(length(c2f)) cells\n    $(length(f2n)) internal faces\n    $(length(b2n)) boundary faces\n    $(length(pts_s)) nodes", verbose)
    return UnstructuredMesh(c2f, c2b, f2n, b2n, pts_s, int_neighbors, bnd_neighbors; kwarg...)
end
