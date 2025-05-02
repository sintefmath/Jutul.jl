"""
    radial_mesh(angles, radii; centerpoint = false, kwarg...)

Create a radial (polar) mesh with the specified angles and radii.

# Arguments
- `angles`: Either an integer specifying the number of angle segments (will
  create a uniform distribution from 0 to 2π), or a vector of angles in radians
  from 0 to 2π in increasing order.
- `radii`: A vector of radial distances from the center.
- `centerpoint`: Boolean indicating whether to include a point at the center of
  the mesh (default: false). When true, the center becomes a node and the mesh
  is divided into radial sectors all the way to the center. When false, the
  innermost radial ring forms a single cell.

# Returns
An `UnstructuredMesh` instance representing the radial mesh.
"""
function radial_mesh(angles, radii; centerpoint = false, kwarg...)
    if angles isa Int
        angles = range(0, 2π, length = angles + 1)
        angles = collect(angles)
    end
    if !(minimum(angles) ≈ 0) || !(maximum(angles) ≈ 2π)
        throw(ArgumentError("Angles must have entries from 0 to 2π (non-inclusive), was $angles"))
    end
    # Remove first entry
    angles = angles[2:end]

    for i in 2:length(angles)
        δ = angles[i] - angles[i-1]
        δ > 0 || throw(ArgumentError("Angles entries must be in increasing order with entries between 0 and 1"))
        abs(δ) > 1e-10 || throw(ArgumentError("Difference between entries below tolerance $tol for entry $i ($(abs(δ)))."))
    end

    nangle = length(angles)
    nangle > 2 || "Number of angles must be greater than 2" || throw(ArgumentError("Number of angles must be greater than 2"))

    Pt_t = SVector{2, Float64}

    point_lookup = OrderedDict{Tuple{Int, Int}, Int}()
    node_points = Pt_t[]
    for (i, r) in enumerate(radii)
        for (j, a) in enumerate(angles)
            x = r * cos(a) - 0.5
            y = r * sin(a) - 0.5
            push!(node_points, Pt_t(x, y))
            point_lookup[(i, j)] = length(node_points)
        end
    end

    if centerpoint
        # Add centerpoint
        x = 0.0 - 0.5
        y = 0.0 - 0.5
        push!(node_points, Pt_t(x, y))
        point_lookup[(0, 0)] = length(node_points)
    end

    # Cells 
    Nradii = length(radii)
    Nangles = length(angles)

    get_radial_cell_index(radius_i, angle_i) = radial_mesh_cell_index(radius_i, angle_i, Nradii, Nangles, centerpoint)

    num_cells = get_radial_cell_index(Nradii, Nangles)
    if centerpoint
        @assert num_cells == Nangles*Nradii
    else
        @assert num_cells == Nangles*(Nradii - 1) + 1
    end

    # Make life easier by these intermediate arrays
    cells_to_faces = Vector{Int}[]
    cells_to_boundary = Vector{Int}[]
    for _ in 1:num_cells
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

    function next_periodic(i::Int, N::Int)
        if i == N
            out = 1
        else
            out = i + 1
        end
        return out
    end

    function prev_periodic(i::Int, N::Int)
        if i == 1
            out = N
        else
            out = i - 1
        end
        return out
    end

    function add_face!(p1, p2, c1, c2)
        @assert p1 > 0
        @assert p2 > 0
        @assert c1 > 0
        @assert c2 > 0
        @assert c1 != c2 "Cells must be different (c1: $c1, c2: $c2)"
        faceno = length(face_node_pos)
        push!(cells_to_faces[c1], faceno)
        push!(cells_to_faces[c2], faceno)
        push!(neighbors, (c1, c2))
        push!(face_node_pos, face_node_pos[end] + 2)
        # Positive normal direction outwards from center
        push!(face_nodes, p1, p2)
    end

    function add_bnd_face!(p1, p2, c)
        @assert p1 > 0
        @assert p2 > 0
        @assert c > 0
        faceno = length(bnd_node_pos)
        push!(cells_to_boundary[c], faceno)
        push!(bnd_cells, c)
        push!(bnd_node_pos, bnd_node_pos[end] + 2)
        # Positive normal direction outwards from center
        push!(bnd_nodes, p1, p2)
    end

    # Faces with constant radius (going in circles around the center)
    for radius_i in 1:Nradii
        for angle_i in 1:Nangles
            angle_i_next = next_periodic(angle_i, Nangles)
            p1 = point_lookup[(radius_i, angle_i)]
            p2 = point_lookup[(radius_i, angle_i_next)]

            if radius_i == Nradii
                # Boundary layer on the outside
                # TODO: Check orientation
                c = get_radial_cell_index(radius_i, angle_i)
                add_bnd_face!(p1, p2, c)
            else
                # Regular internal layer
                if radius_i == 1 && !centerpoint
                    c1 = get_radial_cell_index(radius_i, 1)
                else
                    c1 = get_radial_cell_index(radius_i, angle_i)
                end
                c2 = get_radial_cell_index(radius_i+1, angle_i)
                add_face!(p1, p2, c1, c2)
            end
        end
    end

    # Face with constant angle (tangential to the circles, radiating from the center)
    for radius_i in 1:Nradii
        for angle_i in 1:Nangles
            angle_i_next = prev_periodic(angle_i, Nangles)
            if radius_i == 1
                # Center part
                if centerpoint
                    p1 = point_lookup[(0, 0)]
                    p2 = point_lookup[(radius_i, angle_i)]
                    c1 = get_radial_cell_index(radius_i, angle_i)
                    c2 = get_radial_cell_index(radius_i, angle_i_next)
                    add_face!(p1, p2, c1, c2)
                end
            else
                # Regular internal layer
                c1 = get_radial_cell_index(radius_i, angle_i)
                c2 = get_radial_cell_index(radius_i, angle_i_next)
                p1 = point_lookup[(radius_i-1, angle_i)]
                p2 = point_lookup[(radius_i, angle_i)]
                add_face!(p1, p2, c1, c2)
            end
        end
    end


    cells_faces, cells_facepos = cellmap_to_posmap(cells_to_faces)
    boundary_cells_faces, boundary_cells_facepos = cellmap_to_posmap(cells_to_boundary)

    m = UnstructuredMesh(
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
        structure = CartesianIndex(Nangles, Nradii),
        kwarg...
    )
    return m
end

function radial_mesh_cell_index(radial_index, angular_index, Nradii, Nangles, centerpoint::Bool)
    @assert radial_index >= 1
    @assert angular_index >= 1
    @assert radial_index <= Nradii "Radius index: $radial_index exceeded $(Nradii)"
    @assert angular_index <= Nangles "Angle index: $angular_index exceeded $(Nangles)"
    if radial_index == 1 && !centerpoint
        @assert angular_index == 1 "Center only at the first angle"
        ix = 1
    else
        if centerpoint
            ix = angular_index + (radial_index - 1) * Nangles
        else
            ix = angular_index + max(radial_index - 2, 0) * Nangles + 1
        end
    end
    return ix
end
