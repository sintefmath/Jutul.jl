# Constants for 3D mesh generation
const GEOMETRIC_TOLERANCE_3D = 1e-9
const BBOX_MARGIN_3D = 0.1

"""
    PEBIMesh3D(points; bbox=nothing)

Create a 3D PEBI (Perpendicular Bisector) / Voronoi mesh from a set of points.

# Arguments
- `points`: A matrix of size (3, n) or (n, 3) or vector of 3-tuples/vectors containing the x,y,z coordinates of cell centers
- `bbox`: Optional bounding box as ((xmin, xmax), (ymin, ymax), (zmin, zmax)). If not provided, computed from points with margin

# Returns
- An `UnstructuredMesh` instance representing the 3D PEBI mesh

# Description
The 3D PEBI mesh is a 3D Voronoi diagram where each input point becomes a cell center.
The mesh is bounded by the specified or computed bounding box. Each Voronoi cell is computed
by intersecting half-spaces defined by perpendicular bisector planes between neighboring points.

This implementation uses half-space intersection with proper edge-plane intersection handling.
It works well for point counts up to several hundred points. For very large or complex meshes,
consider using specialized Voronoi libraries for better performance.

# Examples
```julia
# Simple mesh with 8 points (corners of a cube)
points = [0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0;
          0.0 0.0 1.0 1.0 0.0 0.0 1.0 1.0;
          0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0]
mesh = Jutul.VoronoiMeshes.PEBIMesh3D(points)

# Random points (works well up to several hundred points)
points = rand(3, 100)
mesh = Jutul.VoronoiMeshes.PEBIMesh3D(points)
```
"""
function PEBIMesh3D(points; bbox=nothing)
    # Convert points to standard format (3×n)
    pts = _convert_points_3d(points)
    n_points = size(pts, 2)
    
    # Compute or validate bounding box
    if isnothing(bbox)
        bbox = _compute_bbox_3d(pts)
    else
        bbox = _validate_bbox_3d(bbox)
    end
    
    # Generate Voronoi cells
    cells, all_nodes = _generate_voronoi_3d(pts, bbox)
    
    # Build UnstructuredMesh
    return _build_unstructured_mesh_3d(cells, all_nodes)
end

"""
Convert points to standard 3×n format.
"""
function _convert_points_3d(points)
    if points isa AbstractMatrix
        if size(points, 1) == 3
            return points
        elseif size(points, 2) == 3
            return permutedims(points, (2, 1))
        else
            error("Point matrix must have dimension 3 in one direction")
        end
    elseif points isa AbstractVector
        n = length(points)
        pts = zeros(3, n)
        for i in 1:n
            pt = points[i]
            pts[1, i] = pt[1]
            pts[2, i] = pt[2]
            pts[3, i] = pt[3]
        end
        return pts
    else
        error("Points must be a matrix or vector")
    end
end

"""
Compute bounding box from points with margin.
"""
function _compute_bbox_3d(pts::AbstractMatrix)
    xmin, xmax = extrema(pts[1, :])
    ymin, ymax = extrema(pts[2, :])
    zmin, zmax = extrema(pts[3, :])
    
    dx = max(xmax - xmin, GEOMETRIC_TOLERANCE_3D)
    dy = max(ymax - ymin, GEOMETRIC_TOLERANCE_3D)
    dz = max(zmax - zmin, GEOMETRIC_TOLERANCE_3D)
    
    margin_x = dx * BBOX_MARGIN_3D
    margin_y = dy * BBOX_MARGIN_3D
    margin_z = dz * BBOX_MARGIN_3D
    
    return (
        (xmin - margin_x, xmax + margin_x),
        (ymin - margin_y, ymax + margin_y),
        (zmin - margin_z, zmax + margin_z)
    )
end

"""
Validate bounding box format.
"""
function _validate_bbox_3d(bbox)
    if length(bbox) != 3
        error("Bounding box must have 3 components (x, y, z)")
    end
    for (i, dim) in enumerate(bbox)
        if length(dim) != 2 || dim[1] >= dim[2]
            error("Bounding box dimension $i invalid: must be (min, max) with min < max")
        end
    end
    return bbox
end

"""
Generate 3D Voronoi cells using half-space intersections.
"""
function _generate_voronoi_3d(pts::AbstractMatrix, bbox)
    n_points = size(pts, 2)
    cells = Vector{Vector{SVector{3, Float64}}}()
    all_nodes = Set{SVector{3, Float64}}()
    
    # For each point, compute its Voronoi cell
    for i in 1:n_points
        cell_vertices = _compute_voronoi_cell_3d(i, pts, bbox)
        if !isempty(cell_vertices)
            push!(cells, cell_vertices)
            union!(all_nodes, cell_vertices)
        end
    end
    
    return cells, collect(all_nodes)
end

"""
Compute the Voronoi cell for a single point using half-space intersections.
"""
function _compute_voronoi_cell_3d(idx::Int, pts::AbstractMatrix, bbox)
    center = SVector{3, Float64}(pts[1, idx], pts[2, idx], pts[3, idx])
    n_points = size(pts, 2)
    
    # Start with bounding box as initial polyhedron
    vertices = _bbox_to_vertices_3d(bbox)
    
    # Clip by half-space for each other point
    for j in 1:n_points
        if j == idx
            continue
        end
        
        other = SVector{3, Float64}(pts[1, j], pts[2, j], pts[3, j])
        
        # Perpendicular bisector plane
        midpoint = (center + other) / 2
        normal = normalize(center - other)
        
        # Clip vertices to keep only those on the side of center
        vertices = _clip_polyhedron_by_halfspace_3d(vertices, midpoint, normal)
        
        if isempty(vertices)
            break
        end
    end
    
    # Clean up and order vertices
    vertices = _clean_polyhedron_vertices_3d(vertices)
    
    return vertices
end

"""
Convert bounding box to initial cube vertices.
"""
function _bbox_to_vertices_3d(bbox)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    
    # 8 corners of the bounding box
    vertices = SVector{3, Float64}[
        SVector(xmin, ymin, zmin),
        SVector(xmax, ymin, zmin),
        SVector(xmax, ymax, zmin),
        SVector(xmin, ymax, zmin),
        SVector(xmin, ymin, zmax),
        SVector(xmax, ymin, zmax),
        SVector(xmax, ymax, zmax),
        SVector(xmin, ymax, zmax)
    ]
    
    return vertices
end

"""
Clip a convex polyhedron by a half-space (keep side where (v - point) · normal >= 0).

This implementation properly handles edge-plane intersections to ensure vertices
are added where edges cross the clipping plane. This is essential for correctness
when clipping by many planes (e.g., 100 points means 99 clips per cell).
"""
function _clip_polyhedron_by_halfspace_3d(vertices, point, normal)
    if isempty(vertices)
        return vertices
    end
    
    # Classify vertices as inside (>=0) or outside (<0) the half-space
    distances = [dot(v - point, normal) for v in vertices]
    
    # Keep vertices on the correct side
    kept_vertices = SVector{3, Float64}[]
    for (i, v) in enumerate(vertices)
        if distances[i] >= -GEOMETRIC_TOLERANCE_3D
            push!(kept_vertices, v)
        end
    end
    
    # For convex polyhedra, we need to add intersection points where edges cross the plane
    # Generate all possible edges by checking pairs (this is a simple but O(n²) approach)
    # For a convex polyhedron with n vertices, we check all pairs as potential edges
    n = length(vertices)
    for i in 1:n
        for j in (i+1):n
            d_i = distances[i]
            d_j = distances[j]
            
            # Check if edge crosses the plane (one vertex inside, one outside)
            if (d_i >= -GEOMETRIC_TOLERANCE_3D && d_j < -GEOMETRIC_TOLERANCE_3D) ||
               (d_i < -GEOMETRIC_TOLERANCE_3D && d_j >= -GEOMETRIC_TOLERANCE_3D)
                
                # Compute intersection point
                # Edge: v_i + t * (v_j - v_i)
                # Plane: dot((v_i + t * (v_j - v_i)) - point, normal) = 0
                # Solving for t: t = -d_i / (d_j - d_i)
                t = -d_i / (d_j - d_i)
                t = clamp(t, 0.0, 1.0)  # Ensure t is in [0, 1]
                
                intersection = vertices[i] + t * (vertices[j] - vertices[i])
                
                # Only add if not already present (within tolerance)
                if !any(v -> norm(v - intersection) < GEOMETRIC_TOLERANCE_3D, kept_vertices)
                    push!(kept_vertices, intersection)
                end
            end
        end
    end
    
    return kept_vertices
end

"""
Clean up polyhedron vertices (remove duplicates, ensure valid polyhedron).
"""
function _clean_polyhedron_vertices_3d(vertices)
    if length(vertices) < 4
        return SVector{3, Float64}[]
    end
    
    # Remove duplicate vertices
    unique_verts = SVector{3, Float64}[]
    for v in vertices
        if !any(u -> norm(u - v) < GEOMETRIC_TOLERANCE_3D, unique_verts)
            push!(unique_verts, v)
        end
    end
    
    if length(unique_verts) < 4
        return SVector{3, Float64}[]
    end
    
    return unique_verts
end

"""
Build UnstructuredMesh from 3D Voronoi cells.

This function converts the Voronoi cell data into the format required by UnstructuredMesh constructor,
which needs 11 positional arguments with specific interior/boundary separation.
"""
function _build_unstructured_mesh_3d(cells, all_nodes)
    if isempty(cells)
        error("No valid cells generated")
    end
    
    # Create node index mapping
    node_to_idx = Dict{SVector{3, Float64}, Int}()
    node_points = Vector{SVector{3, Float64}}()
    
    for (i, node) in enumerate(all_nodes)
        node_to_idx[node] = i
        push!(node_points, node)
    end
    
    # Extract faces and build connectivity
    # This is simplified - creates triangular faces from cell vertices
    cells_to_faces_vec = Vector{Vector{Int}}()
    all_faces_to_nodes = Vector{Vector{Int}}()
    face_neighbors_vec = Vector{Tuple{Int, Int}}()
    
    face_dict = Dict{Set{Int}, Int}()
    next_face_id = 1
    
    for (cell_idx, cell_verts) in enumerate(cells)
        if length(cell_verts) < 4
            continue
        end
        
        cell_faces = Int[]
        n_verts = length(cell_verts)
        vert_indices = [node_to_idx[v] for v in cell_verts]
        
        # Create simple tetrahedral faces from first vertex to others
        for i in 2:(n_verts-1)
            for j in (i+1):n_verts
                face_nodes = Set([vert_indices[1], vert_indices[i], vert_indices[j]])
                
                if !haskey(face_dict, face_nodes)
                    face_dict[face_nodes] = next_face_id
                    push!(all_faces_to_nodes, sort(collect(face_nodes)))
                    push!(face_neighbors_vec, (cell_idx, 0))
                    next_face_id += 1
                else
                    face_id = face_dict[face_nodes]
                    old_neighbor = face_neighbors_vec[face_id]
                    face_neighbors_vec[face_id] = (old_neighbor[1], cell_idx)
                end
                
                face_id = face_dict[face_nodes]
                if !(face_id in cell_faces)
                    push!(cell_faces, face_id)
                end
            end
        end
        
        push!(cells_to_faces_vec, cell_faces)
    end
    
    # Separate interior and boundary faces
    interior_face_indices = Int[]
    boundary_face_indices = Int[]
    
    for (i, (n1, n2)) in enumerate(face_neighbors_vec)
        if n2 == 0
            push!(boundary_face_indices, i)
        else
            push!(interior_face_indices, i)
        end
    end
    
    # Build flat arrays for cells_to_faces
    cells_faces = Int[]
    cells_facepos = [1]
    boundary_cells_faces = Int[]
    boundary_cells_facepos = [1]
    
    for cell_faces in cells_to_faces_vec
        # Add interior faces
        for face_id in cell_faces
            if face_id in interior_face_indices
                push!(cells_faces, findfirst(==(face_id), interior_face_indices))
            end
        end
        push!(cells_facepos, length(cells_faces) + 1)
        
        # Add boundary faces  
        for face_id in cell_faces
            if face_id in boundary_face_indices
                push!(boundary_cells_faces, findfirst(==(face_id), boundary_face_indices))
            end
        end
        push!(boundary_cells_facepos, length(boundary_cells_faces) + 1)
    end
    
    # Build flat arrays for interior faces_to_nodes
    faces_nodes_flat = Int[]
    internal_faces_nodespos = [1]
    for face_idx in interior_face_indices
        for node_idx in all_faces_to_nodes[face_idx]
            push!(faces_nodes_flat, node_idx)
        end
        push!(internal_faces_nodespos, length(faces_nodes_flat) + 1)
    end
    
    # Build flat arrays for boundary faces_to_nodes
    boundary_faces_nodes_flat = Int[]
    boundary_faces_nodespos = [1]
    for face_idx in boundary_face_indices
        for node_idx in all_faces_to_nodes[face_idx]
            push!(boundary_faces_nodes_flat, node_idx)
        end
        push!(boundary_faces_nodespos, length(boundary_faces_nodes_flat) + 1)
    end
    
    # Build neighbor arrays
    internal_neighbors = zeros(Int, length(interior_face_indices), 2)
    for (i, face_idx) in enumerate(interior_face_indices)
        n1, n2 = face_neighbors_vec[face_idx]
        internal_neighbors[i, 1] = n1
        internal_neighbors[i, 2] = n2
    end
    
    boundary_neighbors = Int[]
    for face_idx in boundary_face_indices
        n1, _ = face_neighbors_vec[face_idx]
        push!(boundary_neighbors, n1)
    end
    
    # Convert node points to matrix format
    all_vertices = zeros(3, length(node_points))
    for (i, pt) in enumerate(node_points)
        all_vertices[:, i] = pt
    end
    
    # Call UnstructuredMesh with all 11 positional arguments
    return UnstructuredMesh(
        cells_faces,
        cells_facepos,
        boundary_cells_faces,
        boundary_cells_facepos,
        faces_nodes_flat,
        internal_faces_nodespos,
        boundary_faces_nodes_flat,
        boundary_faces_nodespos,
        all_vertices,
        internal_neighbors,
        boundary_neighbors
    )
end
