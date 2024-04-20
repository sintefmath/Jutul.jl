function number_of_boundary_faces(g::UnstructuredMesh)
    return length(g.boundary_faces.faces_to_nodes)
end

function number_of_cells(g::UnstructuredMesh)
    return length(g.faces.cells_to_faces)
end

function number_of_faces(g::UnstructuredMesh)
    return length(g.faces.faces_to_nodes)
end

function Base.show(io::IO, t::MIME"text/plain", g::UnstructuredMesh)
    nc = number_of_cells(g)
    nf = number_of_faces(g)
    nb = number_of_boundary_faces(g)
    print(io, "UnstructuredMesh with $nc cells, $nf faces and $nb boundary faces")
end

export extract_submesh

function extract_submesh(g, arg...; kwarg...)
    extract_submesh(UnstructuredMesh(g), arg...; kwarg...)
end


"""
    extract_submesh(g::UnstructuredMesh, cells)

Extract a subgrid for a given mesh and a iterable of `cells` to keep.
"""
function extract_submesh(g::UnstructuredMesh, cells)
    function add_to_indexmap!(vals, pos, iterable)
        n = length(iterable)
        for i in iterable
            push!(vals, i)
        end
        push!(pos, pos[end]+n)
    end

    nf = number_of_faces(g)
    nc = number_of_cells(g)
    # Create renumerators for cells and nodes
    active_cells = IndexRenumerator(cells)
    active_nodes = IndexRenumerator()
    active_faces = IndexRenumerator()
    # Interior faces now converted to boundary faces Two-part index: Bool
    # indicating if it is an interior face converted to boundary and the second
    # is the index of either the interior or boundary face being kept.
    active_bnd = IndexRenumerator(Tuple{Bool, Int})

    # Go over interior and exterior faces and add them to their new lists
    # depending on where they are going to be found in the new neighbor list.
    new_face_nodespos = Int[1]
    new_face_nodes = Int[]

    new_boundary_nodespos = Int[1]
    new_boundary_nodes = Int[]

    new_boundary_cells = Int[]
    new_neighbors = Tuple{Int, Int}[]

    # First we handle the interior faces
    f2n = g.faces.faces_to_nodes
    for face in 1:length(f2n)
        nodes = f2n[face]
        l, r = g.faces.neighbors[face]
        l_act = l in active_cells
        r_act = r in active_cells
        if !l_act && !r_act
            # Do nothing. Face has been removed.
            continue
        end

        if l_act && r_act
            push!(new_neighbors, (active_cells[l], active_cells[r]))
            add_to_indexmap!(new_face_nodes, new_face_nodespos, nodes)
            active_faces(face)
        else
            if l_act
                bnd_cell = active_cells[l]
            else
                @assert r_act
                # Reverse the nodes here if boundary cell == r so that the
                # normal is flipped.
                reverse!(nodes)
                bnd_cell = active_cells[r]
            end
            push!(new_boundary_cells, bnd_cell)
            add_to_indexmap!(new_boundary_nodes, new_boundary_nodespos, nodes)
            active_bnd((true, face))
        end
        for n in nodes
            active_nodes(n)
        end
    end
    # Then the boundary faces
    b2n = g.boundary_faces.faces_to_nodes
    for bf in 1:length(b2n)
        nodes = b2n[bf]
        cell = g.boundary_faces.neighbors[bf]
        if cell in active_cells
            push!(new_boundary_cells, active_cells[cell])
            add_to_indexmap!(new_boundary_nodes, new_boundary_nodespos, nodes)
            for n in nodes
                active_nodes(n)
            end
            active_bnd((false, bf))
        end
    end
    # Remap the nodes
    renumber!(new_face_nodes, active_nodes)
    renumber!(new_boundary_nodes, active_nodes)
    node_points = g.node_points[indices(active_nodes)]

    # We can now remap the cell -> faces map
    new_cells_facepos = Int[1]
    new_cells_faces = Int[]
    # And the cell -> boundary map
    boundary_cells_facepos = Int[1]
    boundary_cells_faces = Int[]

    for cell in cells
        num_bnd = 0
        num_int = 0
        for face in g.faces.cells_to_faces[cell]
            if face in active_faces
                faceix = active_faces[face]
                push!(new_cells_faces, faceix)
                num_int += 1
            else
                @assert (true, face) in active_bnd
                # Is this wrong?
                faceix = active_bnd[(true, face)]
                push!(boundary_cells_faces, faceix)
                num_bnd += 1
            end
        end
        for face in g.boundary_faces.cells_to_faces[cell]
            @assert (false, face) in active_bnd
            faceix = active_bnd[(false, face)]
            push!(boundary_cells_faces, faceix)
            num_bnd += 1
        end
        push!(new_cells_facepos, new_cells_facepos[end]+num_int)
        push!(boundary_cells_facepos, boundary_cells_facepos[end]+num_bnd)
    end

    return UnstructuredMesh(
        new_cells_faces,
        new_cells_facepos,
        boundary_cells_faces,
        boundary_cells_facepos,
        new_face_nodes,
        new_face_nodespos,
        new_boundary_nodes,
        new_boundary_nodespos,
        node_points,
        new_neighbors,
        new_boundary_cells,
        structure = g.structure,
        cell_map = cells
    )
end
