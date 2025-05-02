function trajectory_to_points(trajectory::Matrix{Float64})
    N = size(trajectory, 2)
    @assert N in (2, 3) "2D/3D matrices are supported."
    return collect(vec(reinterpret(SVector{N, Float64}, collect(trajectory'))))
end

function trajectory_to_points(x::AbstractVector{SVector{N, Float64}}) where N
    return x
end


"""
    find_enclosing_cells(G, traj; geometry = tpfv_geometry(G), n = 25)

Find the cell indices of cells in the mesh `G` that are intersected by a given
trajectory `traj`. `traj` can be either a matrix with equal number of columns as
dimensions in G (i.e. three columns for 3D) or a `Vector` of `SVector` instances
with the same length.

The optional argument `geometry` is used to define the centroids and normals
used in the calculations. You can precompute this if you need to perform many
searches. The keyword argument `n` can be used to set the number of
discretizations in each segment.

`use_boundary` is by default set to `false`. If set to true, the boundary faces
of cells are treated more rigorously when picking exactly what cells are cut by
a trajectory, but this requires that the boundary normals are oriented outwards,
which is currently not the case for all meshes from downstream packages.

`limit_box` speeds up the search by limiting the search to the minimal bounding
box that contains both the trajectory and the mesh. This can be turned off by
passing `false`. There should be no difference in the cells tagged by changing
this option.
"""
function find_enclosing_cells(G, traj;
        geometry = missing,
        n = 25,
        use_boundary = false,
        atol = 0.01,
        limit_box = true
    )
    G = UnstructuredMesh(G)
    if ismissing(geometry)
        geometry = tpfv_geometry(G)
    end
    pts = trajectory_to_points(traj)
    length(pts) > 1 || throw(ArgumentError("Trajectory must have at least two points."))
    T = eltype(pts)
    # Refine the segments
    new_pts = T[]
    for i in 1:(length(pts)-1)
        pt_start = pts[i]
        pt_end = pts[i+1]
        for pt in range(pt_start, pt_end, n)
            push!(new_pts, pt)
        end
    end
    pts = new_pts
    # Turn geometry matrices into vectors of vectors
    normals = vec(reinterpret(T, geometry.normals))
    face_centroids = vec(reinterpret(T, geometry.face_centroids))
    cell_centroids = vec(reinterpret(T, geometry.cell_centroids))

    boundary_centroids = vec(reinterpret(T, geometry.boundary_centroids))
    if use_boundary
        boundary_normals = vec(reinterpret(T, geometry.boundary_normals))
    else
        boundary_normals = boundary_centroids .- cell_centroids[G.boundary_faces.neighbors]
        for i in eachindex(boundary_normals)
            boundary_normals[i] /= norm(boundary_normals[i], 2)
        end
    end

    if limit_box
        # Create a minimal bounding box of the points of the trajectory and grid
        # itself. Cells and points outside this box will not be considered.
        D = dim(G)
        function bounding_box(pts)
            lo = fill(Inf, D)
            hi = fill(-Inf, D)
            for pt in pts
                for i in 1:D
                    lo[i] = min(lo[i], pt[i])
                    hi[i] = max(hi[i], pt[i])
                end
            end
            return (lo, hi)
        end
        lo_g, hi_g = bounding_box(G.node_points)
        lo_t, hi_t = bounding_box(pts)
        # Find the intersection of the bounding boxes
        low_bb = max.(lo_g, lo_t)
        high_bb = min.(hi_g, hi_t)

        inside_bb(x) = point_in_bounding_box(x, low_bb, high_bb, atol = atol)
        pts = filter(inside_bb, pts)
        # Find cells via their nodes - if any node is inside BB we consider the cell
        cells = cells_inside_bounding_box(G, low_bb, high_bb, atol = atol)
    else
        cells = 1:number_of_cells(G)
    end
    # Start search near middle of trajectory
    mean_pt = mean(pts)
    cells_by_dist = sort(cells, by = cell -> norm(cell_centroids[cell] - mean_pt, 2))

    intersected_cells = Int[]
    lengths = Float64
    for pt in pts
        ix = find_enclosing_cell(G, pt, normals, face_centroids, boundary_normals, boundary_centroids, cells_by_dist)
        if !isnothing(ix)
            push!(intersected_cells, ix)
        end
    end
    return unique!(intersected_cells)
end

function point_in_bounding_box(pt, low_bb, high_bb; atol::Float64 = 0.01)
    N = length(pt)
    N == length(low_bb) == length(high_bb) || throw(ArgumentError("Dimensions must match."))
    for i in 1:N
        pt_i = pt[i]
        if pt_i < low_bb[i] - atol
            return false
        end
        if pt_i > high_bb[i] + atol
            return false
        end
    end
    return true
end

"""
    find_enclosing_cell(G::UnstructuredMesh{D}, pt::SVector{D, T},
        normals::AbstractVector{SVector{D, T}},
        face_centroids::AbstractVector{SVector{D, T}},
        boundary_normals::AbstractVector{SVector{D, T}},
        boundary_centroids::AbstractVector{SVector{D, T}},
        cells = 1:number_of_cells(G)
    ) where {D, T}

Find enclosing cell of a point. This can be a bit expensive for larger meshes.
Recommended to use the more high level `find_enclosing_cells` instead.
"""
function find_enclosing_cell(G::UnstructuredMesh{D}, pt::SVector{D, T},
        normals::AbstractVector{SVector{D, T}},
        face_centroids::AbstractVector{SVector{D, T}},
        boundary_normals::AbstractVector{SVector{D, T}},
        boundary_centroids::AbstractVector{SVector{D, T}},
        cells = 1:number_of_cells(G)
    ) where {D, T}
    inside_normal(pt, normal, centroid) = dot(normal, pt - centroid) <= 0
    for cell in cells
        inside = true
        for face in G.faces.cells_to_faces[cell]
            if G.faces.neighbors[face][1] == cell
                sgn = 1
            else
                sgn = -1
            end
            normal = sgn*normals[face]
            center = face_centroids[face]
            inside = inside && inside_normal(pt, normal, center)
            if !inside
                break
            end
        end
        if !inside
            continue
        end

        for bface in G.boundary_faces.cells_to_faces[cell]
            normal = boundary_normals[bface]
            center = boundary_centroids[bface]
            inside = inside && inside_normal(pt, normal, center)
            if !inside
                break
            end
        end
        # A final check to see if the point is inside the bounding box of the
        # cell. This is not strictly necessary, but can be useful for some
        # degenerate geometries.
        if inside && point_inside_cell_bounding_box(G, cell, pt)
            return cell
        end
    end
    return nothing
end

function point_inside_cell_bounding_box(G::UnstructuredMesh, cell, pt::SVector{D, T}; atol = 0.0) where {D, T}
    bb_low = zero(SVector{D, T}) .+ Inf
    bb_high = zero(SVector{D, T}) .- Inf
    for face in G.faces.cells_to_faces[cell]
        for pt in G.faces.faces_to_nodes[face]
            bb_low = min.(bb_low, G.node_points[pt])
            bb_high = max.(bb_high, G.node_points[pt])
        end
    end
    for bface in G.boundary_faces.cells_to_faces[cell]
        for pt in G.boundary_faces.faces_to_nodes[bface]
            bb_low = min.(bb_low, G.node_points[pt])
            bb_high = max.(bb_high, G.node_points[pt])
        end
    end
    return point_in_bounding_box(pt, bb_low, bb_high, atol = atol)
end

"""
    cells_inside_bounding_box(G::UnstructuredMesh, low_bb, high_bb; algorithm = :box, atol = 0.01)


"""
function cells_inside_bounding_box(G::UnstructuredMesh, low_bb, high_bb; algorithm = :box, atol = 0.01)
    D = dim(G)
    length(low_bb) == length(high_bb) == D || throw(ArgumentError("Dimensions of bounding box must match with grid dimension $D."))
    nodes = G.node_points
    cells = Int[]

    function bb_overlap(A, B)
        Amin, Amax = A
        Bmin, Bmax = B
        return Amax >= Bmin && Bmax >= Amin
    end

    if algorithm == :nodal
        # Check if any node is inside the bounding box
        node_is_active = fill(false, length(nodes))
        for (i, node) in enumerate(nodes)
            node_is_active[i] = point_in_bounding_box(node, low_bb, high_bb, atol = atol)
        end
        active_faces = Int[]
        for face in 1:length(G.faces.faces_to_nodes)
            for node in G.faces.faces_to_nodes[face]
                if node_is_active[node]
                    push!(active_faces, face)
                    break
                end
            end
        end
        for f in active_faces
            l, r = G.faces.neighbors[f]
            push!(cells, l, r)
        end
    elseif algorithm == :box
        # Check intersection of bounding boxes of cells with the provided bounding box
        low_bb_cell = zeros(D)
        high_bb_cell = zeros(D)
        for cell in 1:number_of_cells(G)
            @. low_bb_cell = Inf
            @. high_bb_cell = -Inf
            for face in G.faces.cells_to_faces[cell]
                for nodeix in G.faces.faces_to_nodes[face]
                    node = nodes[nodeix]
                    low_bb_cell = min.(low_bb_cell, node)
                    high_bb_cell = max.(high_bb_cell, node)
                end
            end
            for face in G.boundary_faces.cells_to_faces[cell]
                for nodeix in G.boundary_faces.faces_to_nodes[face]
                    node = nodes[nodeix]
                    low_bb_cell = min.(low_bb_cell, node)
                    high_bb_cell = max.(high_bb_cell, node)
                end
            end
            inside = true
            for d in 1:D
                dim_overlap = bb_overlap(
                    (low_bb_cell[d], high_bb_cell[d]),
                    (low_bb[d], high_bb[d])
                )
                inside = inside && dim_overlap
            end
            if inside
                push!(cells, cell)
            end
        end
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))

    end
    return unique!(sort!(cells))
end
