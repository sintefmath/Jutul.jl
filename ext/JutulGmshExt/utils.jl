
function add_next!(faces, remap, tags, numpts, offset)
    vals = Int[]
    for j in 1:numpts
        push!(vals, remap[tags[offset + j]])
    end
    push!(faces, vals)
end

function parse_faces(remaps; verbose = false)
    node_remap = remaps.nodes
    face_remap = remaps.faces
    faces = Vector{Int}[]
    for (dim, tag) in gmsh.model.getEntities()
        if dim != 2
            continue
        end
        type = gmsh.model.getType(dim, tag)
        name = gmsh.model.getEntityName(dim, tag)
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
        type = gmsh.model.getType(dim, tag)
        ename = gmsh.model.getEntityName(dim, tag)
        for (etypes, etags, enodetags) in zip(elemTypes, elemTags, elemNodeTags)
            name, dim, order, numv, parv, _ = gmsh.model.mesh.getElementProperties(etypes)
            if name == "Quadrilateral 4"
                numpts = 4
            elseif name == "Triangle 3"
                numpts = 3
            else
                error("Unsupported element type $name for faces.")
            end
            @assert length(enodetags) == numpts*length(etags)
            print_message("Faces: Processing $(length(etags)) tags of type $name", verbose)
            for (i, etag) in enumerate(etags)
                offset = (i-1)*numpts
                add_next!(faces, node_remap, enodetags, numpts, offset)
                face_remap[etag] = length(faces)
            end
            print_message("Added $(length(etags)) faces of type $name with $(length(unique(enodetags))) unique nodes", verbose)
        end
    end
    return faces
end

function get_cell_decomposition(name)
    if name == "Hexahedron 8"
        tris = Tuple{}()
        quads = (
            QUAD_T(0, 4, 7, 3),
            QUAD_T(1, 2, 6, 5),
            QUAD_T(0, 1, 5, 4),
            QUAD_T(2, 3, 7, 6),
            QUAD_T(0, 3, 2, 1),
            QUAD_T(4, 5, 6, 7)
        )
        numpts = 8
    elseif name == "Tetrahedron 4"
        tris = (
            TRI_T(0, 1, 3),
            TRI_T(0, 2, 1),
            TRI_T(0, 3, 2),
            TRI_T(1, 2, 3)
        )
        quads = Tuple{}()
        numpts = 4
    elseif name == "Pyramid 5"
        # TODO: Not really tested.
        tris = (
            TRI_T(0, 1, 4),
            TRI_T(0, 4, 3),
            TRI_T(3, 4, 2),
            TRI_T(1, 2, 4)
        )
        quads = (QUAD_T(0, 3, 2, 1),)
        numpts = 4
    elseif name == "Prism 6"
        # TODO: Not really tested.
        tris = (
            TRI_T(0, 2, 1),
            TRI_T(3, 4, 5)
        )
        quads = (
            QUAD_T(0, 1, 4, 3),
            QUAD_T(0, 3, 5, 2),
            QUAD_T(1, 2, 5, 4)
        )
        numpts = 6
    else
        error("Unsupported element type $name for cells.")
    end
    return (tris, quads, numpts)
end

function print_message(msg, verbose)
    if verbose
        println(msg)
    end
end

function parse_cells(remaps, faces, face_lookup; verbose = false)
    node_remap = remaps.nodes
    face_remap = remaps.faces
    cell_remap = remaps.cells
    cells = Vector{Tuple{Int, Int}}[]
    for (dim, tag) in gmsh.model.getEntities()
        if dim != 3
            continue
        end
        type = gmsh.model.getType(dim, tag)
        name = gmsh.model.getEntityName(dim, tag)
        # Get the mesh elements for the entity (dim, tag):
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
        # * Type and name of the entity:
        type = gmsh.model.getType(dim, tag)
        ename = gmsh.model.getEntityName(dim, tag)
        for (etypes, etags, enodetags) in zip(elemTypes, elemTags, elemNodeTags)
            name, dim, order, numv, parv, _ = gmsh.model.mesh.getElementProperties(etypes)
            tris, quads, numpts = get_cell_decomposition(name)
            print_message("Cells: Processing $(length(etags)) tags of type $name", verbose)
            @assert length(enodetags) == numpts*length(etags)
            nadded = 0
            for (i, etag) in enumerate(etags)
                offset = (i-1)*numpts
                pt_range = (offset+1):(offset+numpts)
                @assert length(pt_range) == numpts
                pts = map(i -> node_remap[enodetags[i]], pt_range)
                cell = Tuple{Int, Int}[]
                for face_t in (tris, quads)
                    for (fno, face) in enumerate(face_t)
                        face_pts = map(i -> pts[i+1], face)
                        face_pts_sorted = sort(face_pts)
                        faceno = get(face_lookup, face_pts_sorted, 0)
                        if faceno == 0
                            nadded += 1
                            push!(faces, face_pts)
                            faceno = length(faces)
                            face_lookup[face_pts_sorted] = faceno
                            sgn = 1
                        else
                            sgn = check_equal_perm(face_pts, faces[faceno]) ? 1 : 2
                        end
                        push!(cell, (faceno, sgn))
                    end
                end
                cell_remap[etag] = length(cells) + 1
                push!(cells, cell)
            end
            print_message("Added $(length(etags)) new cells of type $name and $nadded new faces.", verbose)
        end
    end
    return cells
end

function build_neighbors(cells, faces, face_lookup)
    neighbors = zeros(Int, 2, length(faces))
    for (cellno, cell_to_faces) in enumerate(cells)
        for (face_index, lr) in cell_to_faces
            face_pts = faces[face_index]
            face_pts_sorted = sort(face_pts)
            faceno = face_lookup[face_pts_sorted]
            face_ref = faces[faceno]
            oldn = neighbors[lr, faceno]
            oldn == 0 || error("Cannot overwrite face neighbor for cell $cellno - was already defined as $oldn for index $lr: $(neighbors[:, faceno])")
            neighbors[lr, faceno] = cellno
        end
    end
    return neighbors
end

function generate_face_lookup(faces)
    face_lookup = Dict{Union{QUAD_T, TRI_T}, Int}()

    for (i, face) in enumerate(faces)
        n = length(face)
        if n == 3
            ft = sort(TRI_T(face[1], face[2], face[3]))
        elseif n == 4
            ft = sort(QUAD_T(face[1], face[2], face[3], face[4]))
        else
            error("Unsupported face type with $n nodes, only 3 (for tri) and 4 (for quad) are known.")
        end
        @assert issorted(ft)
        face_lookup[ft] = i
    end
    return face_lookup
end

function split_boundary(neighbors, faces_to_nodes, cells_to_faces, active_ix::Vector{Int}; boundary::Bool)
    remap = OrderedDict{Int, Int}()
    for (i, ix) in enumerate(active_ix)
        remap[ix] = i
    end
    # is_active = [false for _ in eachindex(faces_to_nodes)]
    # is_active[active_ix] .= true
    # Make renumbering here.
    if boundary
        new_neighbors = Int[]
        for ix in active_ix
            l, r = neighbors[:, ix]
            @assert l == 0 || r == 0
            push!(new_neighbors, max(l, r))
        end
    else
        new_neighbors = Tuple{Int, Int}[]
        for ix in active_ix
            l, r = neighbors[:, ix]
            @assert l != 0 && r != 0
            push!(new_neighbors, (l, r))
        end
    end
    new_faces_to_nodes = map(copy, faces_to_nodes[active_ix])
    # Handle cells -> current type of faces
    new_cells_to_faces = Vector{Int}[]
    for cell_to_faces in cells_to_faces
        new_cell = Int[]
        for (face, sgn) in cell_to_faces
            if haskey(remap, face)
                push!(new_cell, remap[face])
            end
        end
        push!(new_cells_to_faces, new_cell)
    end

    return (new_neighbors, new_faces_to_nodes, new_cells_to_faces)
end
