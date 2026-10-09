function normalize_face_nodes(nodes)
    @assert length(nodes) <= 4 "normalize_face_nodes only supports up to four nodes, got $(length(nodes))."
    # Remove zero-length edges without changing the face orientation. A quad
    # with a collapsed edge becomes a triangle rather than a missing face.
    distinct_nodes = Int[]
    for node in nodes
        if isempty(distinct_nodes) || node != last(distinct_nodes)
            push!(distinct_nodes, node)
        end
    end
    if length(distinct_nodes) > 1 && first(distinct_nodes) == last(distinct_nodes)
        pop!(distinct_nodes)
    end
    # A non-adjacent repeated node describes a backtracking polygon, which
    # cannot be repaired by removing zero-length edges.
    if !allunique(distinct_nodes)
        return nothing
    end
    n = length(distinct_nodes)
    if n == 3
        return TRI_T(distinct_nodes)
    elseif n == 4
        return QUAD_T(distinct_nodes)
    else
        # Faces that collapse to a line or point have no surface area.
        return nothing
    end
end

function add_next!(faces, remap, tags, numpts, offset; remove_faces = true)
    vals = Int[]
    for j in 1:numpts
        push!(vals, remap[tags[offset + j]])
    end
    if remove_faces
        face_nodes = normalize_face_nodes(vals)
        if isnothing(face_nodes)
            return false
        end
    else
        face_nodes = vals
    end
    push!(faces, face_nodes)
    return true
end

function parse_faces(remaps; verbose = false, remove_faces = true)
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
            @assert length(enodetags) == numpts * length(etags)
            print_message("Faces: Processing $(length(etags)) tags of type $name", verbose)
            nadded = 0
            for (i, etag) in enumerate(etags)
                offset = (i - 1) * numpts
                if add_next!(faces, node_remap, enodetags, numpts, offset; remove_faces = remove_faces)
                    face_remap[etag] = length(faces)
                    nadded += 1
                end
            end
            print_message("Added $nadded faces of type $name with $(length(unique(enodetags))) unique nodes", verbose)
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
            QUAD_T(4, 5, 6, 7),
        )
        numpts = 8
    elseif name == "Tetrahedron 4"
        tris = (
            TRI_T(0, 1, 3),
            TRI_T(0, 2, 1),
            TRI_T(0, 3, 2),
            TRI_T(1, 2, 3),
        )
        quads = Tuple{}()
        numpts = 4
    elseif name == "Pyramid 5"
        # TODO: Not really tested.
        tris = (
            TRI_T(0, 1, 4),
            TRI_T(0, 4, 3),
            TRI_T(3, 4, 2),
            TRI_T(1, 2, 4),
        )
        quads = (QUAD_T(0, 3, 2, 1),)
        numpts = 4
    elseif name == "Prism 6"
        # TODO: Not really tested.
        tris = (
            TRI_T(0, 2, 1),
            TRI_T(3, 4, 5),
        )
        quads = (
            QUAD_T(0, 1, 4, 3),
            QUAD_T(0, 3, 5, 2),
            QUAD_T(1, 2, 5, 4),
        )
        numpts = 6
    else
        error("Unsupported element type $name for cells.")
    end
    return (tris, quads, numpts)
end

function print_message(msg, verbose)
    return if verbose
        println(msg)
    end
end

function parse_cells(remaps, faces, face_lookup; verbose = false, remove_faces = true)
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
        for (etypes, etags, enodetags) in zip(elemTypes, elemTags, elemNodeTags)
            name, dim, _, _, _, _ = gmsh.model.mesh.getElementProperties(etypes)
            tris, quads, numpts = get_cell_decomposition(name)
            print_message("Cells: Processing $(length(etags)) tags of type $name", verbose)
            @assert length(enodetags) == numpts * length(etags)
            nadded = 0
            nc_before = length(cells)
            for (i, etag) in enumerate(etags)
                offset = (i - 1) * numpts
                pt_range = (offset + 1):(offset + numpts)
                @assert length(pt_range) == numpts
                pts = map(i -> node_remap[enodetags[i]], pt_range)
                cell = Tuple{Int, Int}[]
                # Optionally remove collapsed edges while preserving orientation.
                cell_face_nodes = Union{TRI_T, QUAD_T}[]
                for face_t in (tris, quads)
                    for face in face_t
                        face_pts = map(i -> pts[i + 1], face)
                        if remove_faces
                            face_pts = normalize_face_nodes(face_pts)
                            if isnothing(face_pts)
                                continue
                            end
                        end
                        push!(cell_face_nodes, face_pts)
                    end
                end
                # Compare sorted node indices to recognize the same face even
                # when its node order is reversed.
                sorted_cell_face_nodes = sort.(cell_face_nodes)
                for (face_pts, face_pts_sorted) in zip(cell_face_nodes, sorted_cell_face_nodes)
                    if remove_faces
                        matching_face_count = count(nodes -> nodes == face_pts_sorted, sorted_cell_face_nodes)
                        # Opposing faces can coincide when a cell pinches out. Skip
                        # both occurrences so they cannot introduce extra neighbors.
                        if matching_face_count > 1
                            continue
                        end
                    end
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
                if remove_faces && isempty(cell)
                    continue
                end
                cell_remap[etag] = length(cells) + 1
                push!(cells, cell)
            end
            nc_added = length(cells) - nc_before
            nc_skipped = length(etags) - nc_added
            print_message("Added $nc_added new cells of type $name and $nadded new faces.", verbose)
            if nc_skipped > 0
                print_message("Skipped $nc_skipped cells without valid faces.", verbose)
            end
        end
    end
    return cells
end

function get_cell_tags()
    tags = Dict{UInt64, Int}()
    cellno = 0
    for (dim, tag) in gmsh.model.getEntities()
        if dim != 3
            continue
        end
        elemTags = gmsh.model.mesh.getElements(dim, tag)[2]
        for etags in elemTags
            for etag in etags
                @assert !haskey(tags, etag)
                cellno += 1
                tags[etag] = cellno
            end
        end
    end
    # @assert sort(collect(values(tags))) == 1:cellno
    return tags
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
    if boundary
        for (i, ix) in enumerate(active_ix)
            # The face normal points from the left cell to the right cell.
            # Reverse it when only the right cell remains, so it points outward.
            if neighbors[1, ix] == 0
                reverse!(new_faces_to_nodes[i])
            end
        end
    end
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
