"""
    mesh = spiral_mesh(10, 3, spacing = [0.0, 0.5, 1.0])

Spiral mesh generator. Generates a spiral mesh in 2D with an optional "spacing"
subdiscretization between each segment.
"""
function spiral_mesh(n_angular_sections = 10, nrotations = 5; spacing = 0, start = 2π, A = 1.0, C = 0.0, kwarg...)
    # Code really does nrotations - 1, so we need to add 1 to the number of rotations
    # to get the correct number of cells in the radial direction.
    nrotations += 1
    spacing = spiral_spacing(spacing)
    spacing_width = length(spacing)-1
    dr = 2π/n_angular_sections
    angular_ranges = range(0.0, 2π - dr, length = n_angular_sections)
    # Define all points first
    Pt_t = SVector{2, Float64}
    points = Pt_t[]
    point_list = Vector{Int}[]
    for (rno, r) in enumerate(angular_ranges)
        local_point_list = Int[]
        if rno == 1
            num_active_rot = nrotations
        else
            num_active_rot = nrotations - 1
        end
        for rot in 1:num_active_rot
            ϕ0 = (rot-1)*2π + start + r
            ϕ1 = rot*2π + start + r
            x0, y0 = spiral_coord(ϕ0, A, C)
            x1, y1 = spiral_coord(ϕ1, A, C)
            δx = x1 - x0
            δy = y1 - y0
            for (sno, spacing) in enumerate(spacing)
                # Only add first point if we are at first rotation
                do_add = (rot == 1 && sno == 1) || sno > 1
                if do_add
                    x = x0 + spacing * δx
                    y = y0 + spacing * δy
                    push!(local_point_list, length(points) + 1)
                    push!(points, Pt_t(x, y))
                end
            end
        end
        push!(point_list, local_point_list)
    end
    num_cells_per_angular_section = spacing_width*(nrotations - 1)
    num_cells = n_angular_sections*num_cells_per_angular_section
    function get_cell_index(radial_index, angular_index)
        @assert radial_index > 0 && radial_index <= num_cells_per_angular_section "radial_index $radial_index out of bounds (1 to $num_cells_per_angular_section for angular = $angular_index)"
        @assert angular_index > 0 && angular_index <= n_angular_sections "angular_index $angular_index out of bounds (1 to $n_angular_sections for radial = $radial_index)"
        cix = num_cells_per_angular_section*(angular_index - 1) + radial_index
        return min(cix, num_cells)
    end

    # Make life easier by these intermediate arrays
    cells_to_faces = Vector{Int}[]
    cells_to_boundary = Vector{Int}[]
    for cell3d in 1:num_cells
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

    # Faces with constant angle (i.e. along one radial line)
    for angular_ix in 1:n_angular_sections
        is_displaced_angle = angular_ix == 1
        local_point_list = point_list[angular_ix]
        n_pts = length(local_point_list)
        for radial_ix in 1:(n_pts - 1)
            node1 = local_point_list[radial_ix]
            node2 = local_point_list[radial_ix + 1]
            if is_displaced_angle
                is_start = radial_ix <= spacing_width
                is_end = radial_ix > (nrotations-1)*spacing_width
            else
                is_start = is_end = false
            end
            is_bnd = is_start || is_end
            if is_bnd
                if is_start
                    c1 = get_cell_index(radial_ix, angular_ix)
                else
                    c1 = get_cell_index(radial_ix - spacing_width, n_angular_sections)
                end
                bfaceno = length(bnd_node_pos)
                push!(cells_to_boundary[c1], bfaceno)
                push!(bnd_cells, c1)
                if is_end
                    push!(bnd_nodes, node2, node1)
                else
                    push!(bnd_nodes, node1, node2)
                end
                push!(bnd_node_pos, bnd_node_pos[end] + 2)
            else
                faceno = length(face_node_pos)
                if is_displaced_angle
                    angular_next = n_angular_sections
                    radial_shift_next = spacing_width
                else
                    angular_next = angular_ix - 1
                    radial_shift_next = 0
                end
                c1 = get_cell_index(radial_ix, angular_ix)
                c2 = get_cell_index(radial_ix - radial_shift_next, angular_next)

                push!(cells_to_faces[c1], faceno)
                push!(cells_to_faces[c2], faceno)
                push!(neighbors, (c1, c2))

                push!(face_node_pos, face_node_pos[end] + 2)
                push!(face_nodes, node1, node2)
            end
        end
    end
    # Faces with constant radius (i.e. connecting two radial lines)
    for right in 1:n_angular_sections
        base_range = 1:(spacing_width*(nrotations-1) + 1)
        lrange = base_range
        rrange = base_range
        if right == 1
            left = right + 1
        elseif right == n_angular_sections
            lrange = (spacing_width+1):(spacing_width*nrotations + 1)
            left = 1
        else
            left = right + 1
        end
        @assert length(lrange) == length(rrange)
        right_point_list = point_list[right]
        left_point_list = point_list[left]
        pointno = 1
        for (ir, il) in zip(rrange, lrange)
            r_node = right_point_list[ir]
            l_node = left_point_list[il]
            is_first_bnd = ir == rrange[1]
            is_last_bnd = ir == rrange[end]
            if is_first_bnd || is_last_bnd
                if is_first_bnd
                    @assert il == lrange[1]
                    push!(bnd_nodes, l_node, r_node)
                    cell = get_cell_index(pointno, right)
                else
                    @assert il == lrange[end]
                    push!(bnd_nodes, r_node, l_node)
                    cell = get_cell_index(pointno-1, right)
                end
                bfaceno = length(bnd_node_pos)
                push!(cells_to_boundary[cell], bfaceno)
                push!(bnd_cells, cell)
                push!(bnd_node_pos, bnd_node_pos[end] + 2)
            else
                c1 = get_cell_index(pointno-1, right)
                c2 = get_cell_index(pointno, right)
                faceno = length(face_node_pos)
                push!(cells_to_faces[c1], faceno)
                push!(cells_to_faces[c2], faceno)
                push!(neighbors, (c1, c2))
                push!(face_node_pos, face_node_pos[end] + 2)
                # Positive normal direction outwards from center
                push!(face_nodes, r_node, l_node)
            end
            pointno += 1
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
        points,
        neighbors,
        bnd_cells;
        structure = CartesianIndex(num_cells_per_angular_section, n_angular_sections),
        kwarg...
    )
    return m
end

function spiral_coord(ϕ, a, c)
    r = a*ϕ + c
    x = r * cos(ϕ)
    y = r * sin(ϕ)
    return (x, y)
end

function spiral_spacing(spacing::Int; tol = 1e-10)
    return spacing = range(0.0, 1.0, length = spacing + 2)
end

function spiral_spacing(spacing::AbstractVector; tol = 1e-10)
    if length(spacing) == 0
        spacing = [0.0, 1.0]
    else
        spacing = copy(spacing)
        if maximum(spacing) < 1
            push!(spacing, 1.0)
        elseif maximum(spacing) > 1
            throw(ArgumentError("Spacing entries must be in range [0, 1]"))
        end
        if minimum(spacing) > 0
            pushfirst!(spacing, 0.0)
        elseif minimum(spacing) < 0
            throw(ArgumentError("Spacing entries must be in range [0, 1]"))
        end
        for i in 2:length(spacing)
            δ = spacing[i] - spacing[i-1]
            δ > 0 || throw(ArgumentError("Spacing entries must be in increasing order with entries between 0 and 1"))
            abs(δ) > tol || throw(ArgumentError("Difference between entries below tolerance $tol for entry $i ($(abs(δ)))."))
        end
    end
    return spacing
end

