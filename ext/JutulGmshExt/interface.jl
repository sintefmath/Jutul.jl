
function Jutul.mesh_from_gmsh(pth; manage_gmsh = true, kwarg...)
    if manage_gmsh
        Gmsh.initialize()
    end
    ext = pth |> splitext |> last
    gmsh.open(pth)
    if lowercase(ext) == ".geo"
        gmsh.model.mesh.generate()
    end
    g = missing
    try
        g = Jutul.mesh_from_gmsh(; kwarg...)
    finally
        if manage_gmsh
            Gmsh.finalize()
        end
    end
    if ismissing(g)
        error("Failed to parse mesh")
    end
    return g
end

function Jutul.mesh_from_gmsh(; verbose = false, kwarg...)
    dim = gmsh.model.getDimension()
    dim == 3 || error("Only 3D models are supported")

    gmsh.model.mesh.removeDuplicateNodes()
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

    c2f = IndirectionMap(int_cells_to_faces)
    c2b = IndirectionMap(bnd_cells_to_faces)
    f2n = IndirectionMap(int_faces_to_nodes)
    b2n = IndirectionMap(bnd_faces_to_nodes)
    print_message("Mesh parsed successfully:\n    $(length(c2f)) cells\n    $(length(f2n)) internal faces\n    $(length(b2n)) boundary faces\n    $(length(pts_s)) nodes", verbose)
    return UnstructuredMesh(c2f, c2b, f2n, b2n, pts_s, int_neighbors, bnd_neighbors; kwarg...)
end
