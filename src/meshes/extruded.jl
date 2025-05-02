
"""
    extrude_mesh(m2d::UnstructuredMesh, nlayers)
    extrude_mesh(m2d::UnstructuredMesh, [1, 2, 5, 10])

Extrude a 2D mesh into a 3D mesh by adding layers of cells in the z-direction.
The number of layers can be specified as an integer or as an array of depths.
The depths must be in increasing order.
"""
function extrude_mesh(m2d::UnstructuredMesh, nlayers::Int; kwarg...)
    depths = collect(range(0.0, 1.0, length = nlayers + 1))
    return extrude_mesh(m2d, depths; kwarg...)
end

function extrude_mesh(m2d::UnstructuredMesh, depths; kwarg...)
    for layer in 2:length(depths)
        if depths[layer] <= depths[layer - 1]
            error("Depths must be in increasing order.")
        end
    end
    nc2d = number_of_cells(m2d)
    nf2d = number_of_faces(m2d)
    nb2d = number_of_boundary_faces(m2d)
    nn2d = length(m2d.node_points)
    dim(m2d) == 2 || error("Mesh must be 2D")
    nz = length(depths) - 1
    nz > 0 || error("Number of layers must be greater than 0")
    flat_faces = Vector{Int}[]
    for cell in 1:nc2d
        push!(flat_faces, cell2d_to_face3d(m2d, cell))
    end
    # Extrude nodes first - easiest part
    Pt3d = SVector{3, Float64}
    node_points = Pt3d[]
    for (i, depth) in enumerate(depths)
        for (j, pt) in enumerate(m2d.node_points)
            push!(node_points, Pt3d(pt[1], pt[2], depth))
        end
    end

    nc3d = nz*nc2d
    # Make life easier by these intermediate arrays
    cells_to_faces = Vector{Int}[]
    cells_to_boundary = Vector{Int}[]
    for cell3d in 1:nc3d
        push!(cells_to_faces, Int[])
        push!(cells_to_boundary, Int[])
    end
    # Face to nodes
    face_nodes = Int[]
    face_node_pos = Int[1]
    bnd_nodes = Int[]
    bnd_node_pos = Int[1]
    # Neighbors
    neighbors = Tuple{Int, Int}[]
    bnd_cells = Int[]

    for layer in 1:nz
        offset_cell = (layer - 1) * nc2d
        offset_pts = (layer - 1) * nn2d
        offset_pts_next = layer * nn2d
        # Add vertical faces - these are faces in the original mesh
        for face in 1:nf2d
            nodes_2d = m2d.faces.faces_to_nodes[face]
            c1, c2 = m2d.faces.neighbors[face]
            # Add to cell
            face = length(face_node_pos)
            push!(cells_to_faces[c1 + offset_cell], face)
            push!(cells_to_faces[c2 + offset_cell], face)
            # Add to face
            @assert length(nodes_2d) == 2
            n1, n2 = nodes_2d
            push!(face_nodes, n1 + offset_pts, n2 + offset_pts, n2 + offset_pts_next, n1 + offset_pts_next)
            push!(face_node_pos, face_node_pos[end] + 4)
            push!(neighbors, (c1 + offset_cell, c2 + offset_cell))
        end
        # Vertical boundary faces - same story
        for bface in 1:nb2d
            nodes_2d = m2d.boundary_faces.faces_to_nodes[bface]
            c = m2d.boundary_faces.neighbors[bface]
            push!(cells_to_boundary[c + offset_cell], length(bnd_node_pos))

            @assert length(nodes_2d) == 2
            n1, n2 = nodes_2d
            push!(bnd_nodes, n1 + offset_pts, n2 + offset_pts, n2 + offset_pts_next, n1 + offset_pts_next)
            push!(bnd_node_pos, bnd_node_pos[end] + 4)
            push!(bnd_cells, c + offset_cell)
        end
    end

    # Interior layers
    for layer in 2:nz
        # Orientation is upwards
        above_offset = (layer - 2) * nc2d
        below_offset = (layer - 1) * nc2d
        offset_pts = (layer - 1) * nn2d

        for cell in 1:nc2d
            c_above = below_offset + cell
            c_below = above_offset + cell
            ff = flat_faces[cell]
            for n in ff
                push!(face_nodes, n + offset_pts)
            end
            face = length(face_node_pos)
            push!(cells_to_faces[c_below], face)
            push!(cells_to_faces[c_above], face)

            push!(face_node_pos, face_node_pos[end] + length(ff))
            push!(neighbors, (c_below, c_above))
        end
    end
    for layer in [1, nz+1]
        is_first = layer == 1
        if is_first
            offset = 0
        else
            offset = (nz - 1) * nc2d
        end
        offset_pts = (layer - 1) * nn2d
        for cell in 1:nc2d
            ff = flat_faces[cell]
            if is_first
                ff = reverse(ff)
            end
            for n in ff
                push!(bnd_nodes, n + offset_pts)
            end
            bface = length(bnd_node_pos)
            push!(bnd_node_pos, bnd_node_pos[end] + length(ff))
            push!(bnd_cells, cell + offset)
            push!(cells_to_boundary[cell + offset], bface)
        end
    end

    cells_faces, cells_facepos = cellmap_to_posmap(cells_to_faces)
    boundary_cells_faces, boundary_cells_facepos = cellmap_to_posmap(cells_to_boundary)

    S = m2d.structure
    if S isa CartesianIndex
        S = CartesianIndex(S.I..., nz)
    else
        S = nothing
    end

    extruded_mesh = UnstructuredMesh(
        cells_faces,
        cells_facepos,
        boundary_cells_faces,
        boundary_cells_facepos,
        face_nodes,
        face_node_pos,
        bnd_nodes,
        bnd_node_pos,
        node_points,
        neighbors,
        bnd_cells;
        structure = S,
        kwarg...
    )

    tags_2d = m2d.tags[Cells()]
    for (tag_k, tag) in pairs(tags_2d.tags)
        for (subtag_k, subtag) in pairs(tag)
            subtag3d = similar(subtag, 0)
            for layer in 1:nz
                offset_cell = (layer - 1) * nc2d
                for cell in subtag
                    push!(subtag3d, cell + offset_cell)
                end
            end
            set_mesh_entity_tag!(extruded_mesh, Cells(), tag_k, subtag_k, subtag3d)
        end
    end
    return extruded_mesh
end

function cell2d_to_face3d(m2d, cell)
    dim(m2d) == 2 || error("Only supported for 2D.")
    edges = Tuple{Int, Int}[]
    for face in m2d.faces.cells_to_faces[cell]
        l, r = m2d.faces.neighbors[face]
        fnodes = m2d.faces.faces_to_nodes[face]
        @assert length(fnodes) == 2
        n1, n2 = fnodes
        if l == cell
            push!(edges, (n1, n2))
        else
            @assert r == cell
            push!(edges, (n2, n1))
        end
    end
    for bface in m2d.boundary_faces.cells_to_faces[cell]
        fnodes = m2d.boundary_faces.faces_to_nodes[bface]
        @assert length(fnodes) == 2
        n1, n2 = fnodes
        push!(edges, (n1, n2))
    end
    e_prev = popat!(edges, 1)
    nodes = Int[]
    push!(nodes, e_prev[2])
    while length(edges) > 0
        i = findfirst(e -> isequal(e[1], e_prev[2]), edges)
        @assert !isnothing(i)
        e_prev = popat!(edges, i)
        push!(nodes, e_prev[2])
    end
    # Verify that the normal points upwards
    V_t = SVector{3, Float64}
    pts = m2d.node_points

    p1 = pts[nodes[1]]
    p2 = pts[nodes[2]]
    p3 = pts[nodes[3]]

    v1 = p2 - p1
    v2 = p3 - p2

    A = V_t(v1[1], v1[2], 0.0)
    B = V_t(v2[1], v2[2], 0.0)
    normdir = last(cross(A, B))
    @assert normdir > 0
    return nodes
end
