"""
    PEBIMesh2D(points; constraints=[], bbox=nothing)

Create a 2D PEBI (Perpendicular Bisector) / Voronoi mesh from a set of points.

# Arguments
- `points`: A matrix of size (2, n) or vector of 2-tuples/vectors containing the x,y coordinates of cell centers
- `constraints`: Vector of line constraints, each given as a tuple of two points (p1, p2) where each point is a 2-element vector/tuple
- `bbox`: Optional bounding box as ((xmin, xmax), (ymin, ymax)). If not provided, computed from points with margin

# Returns
- An `UnstructuredMesh` instance representing the PEBI mesh

# Description
The PEBI mesh is a Perpendicular Bisector mesh, also known as a Voronoi diagram.
Each input point becomes a cell center in the resulting mesh. Linear constraints are 
represented as faces in the mesh, ensuring the mesh respects these constraints.
The mesh is bounded by the specified or computed bounding box.

When constraints are specified, cells that intersect the constraint lines are split
into multiple cells (one on each side of the constraint). This means the final mesh
will typically have MORE cells than input points.

# Note on Constraint Endpoints
Constraint endpoints are added as temporary points to structure the mesh. Voronoi cells
centered at these endpoints are created during mesh generation but are filtered out from
the final mesh. This is expected behavior - these temporary cells ensure proper constraint
representation but should not appear as separate cells in the result.

# Examples
```julia
# Simple mesh with 4 points
points = [0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0]
mesh = PEBIMesh2D(points)

# Mesh with a constraint line
points = rand(2, 20)
constraint = ([0.5, 0.0], [0.5, 1.0])  # Vertical line at x=0.5
mesh = PEBIMesh2D(points, constraints=[constraint])
# Result will have > 20 cells due to splitting
```
"""
function PEBIMesh2D(points; constraints=[], bbox=nothing)
    # Convert points to standard format
    pts, npts = _convert_points_2d(points)
    
    # Compute or validate bounding box
    bb = _compute_bbox_2d(pts, bbox)
    
    # Add constraint points to generate refined mesh along constraints
    # This adds constraint endpoints as temporary points to the point set
    pts_with_constraints, constraint_edges, constraint_point_indices = _add_constraint_points_2d(pts, constraints, bb)
    
    # Generate Voronoi diagram - this creates base cells before splitting
    # Cells will be created for ALL points, including the temporary constraint endpoints
    cells_data = _generate_voronoi_2d(pts_with_constraints, bb, constraint_edges)
    
    # Filter out cells that are centered at constraint endpoints
    # These cells should not appear in the final mesh as separate cells.
    # They are temporary artifacts from adding constraint endpoints to structure the mesh.
    # Note: This filtering is EXPECTED and does not remove any user-provided cells.
    # Only cells with center_idx > original_npts are removed (i.e., constraint endpoint cells).
    original_npts = length(pts)
    filtered_cells_data = [cell for (i, cell) in enumerate(cells_data) if cell.center_idx <= original_npts]
    
    # Split cells that are crossed by constraints
    # This is the key change: instead of just clipping, we split into multiple cells
    # Result: The final mesh will have MORE cells than original points due to splitting
    split_cells_data = _split_cells_by_constraints(filtered_cells_data, pts_with_constraints, constraint_edges)
    
    # Build UnstructuredMesh from the Voronoi cells (now including split cells)
    return _build_unstructured_mesh_2d(split_cells_data, pts_with_constraints)
end

"""
Convert various point formats to a vector of SVector{2, Float64}
"""
function _convert_points_2d(points::AbstractMatrix)
    if size(points, 1) == 2
        # Format: 2 x n matrix
        npts = size(points, 2)
        pts = [SVector{2, Float64}(points[1, i], points[2, i]) for i in 1:npts]
    elseif size(points, 2) == 2
        # Format: n x 2 matrix
        npts = size(points, 1)
        pts = [SVector{2, Float64}(points[i, 1], points[i, 2]) for i in 1:npts]
    else
        error("Points matrix must be either (2, n) or (n, 2)")
    end
    return pts, length(pts)
end

function _convert_points_2d(points::AbstractVector)
    # Convert vector of tuples/vectors to SVector
    pts = [SVector{2, Float64}(p[1], p[2]) for p in points]
    return pts, length(pts)
end

"""
Compute or validate bounding box for the mesh
"""
function _compute_bbox_2d(pts, bbox)
    if bbox === nothing
        # Compute bounding box from points with margin
        xmin = minimum(p[1] for p in pts)
        xmax = maximum(p[1] for p in pts)
        ymin = minimum(p[2] for p in pts)
        ymax = maximum(p[2] for p in pts)
        
        # Add 10% margin
        dx = xmax - xmin
        dy = ymax - ymin
        margin_x = max(0.1 * dx, 1e-10)
        margin_y = max(0.1 * dy, 1e-10)
        
        bbox = ((xmin - margin_x, xmax + margin_x), (ymin - margin_y, ymax + margin_y))
    end
    return bbox
end

"""
Add constraint points to ensure constraints are respected in the mesh
"""
function _add_constraint_points_2d(pts, constraints, bb)
    pts_all = copy(pts)
    constraint_edges = []
    constraint_point_indices = []
    
    for constraint in constraints
        p1, p2 = constraint
        p1_sv = SVector{2, Float64}(p1[1], p1[2])
        p2_sv = SVector{2, Float64}(p2[1], p2[2])
        
        # Add constraint endpoints if not already present
        idx1 = findfirst(p -> norm(p - p1_sv) < 1e-10, pts_all)
        if idx1 === nothing
            push!(pts_all, p1_sv)
            idx1 = length(pts_all)
            push!(constraint_point_indices, idx1)
        end
        
        idx2 = findfirst(p -> norm(p - p2_sv) < 1e-10, pts_all)
        if idx2 === nothing
            push!(pts_all, p2_sv)
            idx2 = length(pts_all)
            push!(constraint_point_indices, idx2)
        end
        
        push!(constraint_edges, (idx1, idx2))
    end
    
    return pts_all, constraint_edges, constraint_point_indices
end

"""
Generate Voronoi cells using a simple box-clipping approach
Note: This does NOT clip by constraints - that's handled in splitting phase
"""
function _generate_voronoi_2d(pts, bb, constraint_edges)
    npts = length(pts)
    ((xmin, xmax), (ymin, ymax)) = bb
    
    # For each point, create its Voronoi cell by computing perpendicular bisectors
    # to all other points and intersecting with bounding box
    cells = []
    
    for i in 1:npts
        cell_vertices = _compute_voronoi_cell_2d(i, pts, bb)
        if !isempty(cell_vertices)
            push!(cells, (center_idx=i, vertices=cell_vertices))
        end
    end
    
    return cells
end

"""
Compute the Voronoi cell for a single point (without constraint clipping)
"""
function _compute_voronoi_cell_2d(idx, pts, bb)
    ((xmin, xmax), (ymin, ymax)) = bb
    center = pts[idx]
    
    # Start with bounding box as initial cell
    vertices = [
        SVector{2, Float64}(xmin, ymin),
        SVector{2, Float64}(xmax, ymin),
        SVector{2, Float64}(xmax, ymax),
        SVector{2, Float64}(xmin, ymax)
    ]
    
    # Clip by perpendicular bisectors to all other points
    for j in 1:length(pts)
        if idx == j
            continue
        end
        vertices = _clip_polygon_by_bisector_2d(vertices, center, pts[j])
        if isempty(vertices)
            break
        end
    end
    
    # Remove duplicate vertices and ensure proper ordering
    vertices = _clean_polygon_vertices(vertices)
    
    return vertices
end

"""
Clean polygon vertices by removing duplicates and ensuring CCW ordering
"""
function _clean_polygon_vertices(vertices)
    if isempty(vertices)
        return vertices
    end
    
    # Remove duplicates
    cleaned = SVector{2, Float64}[]
    tol = 1e-10
    
    for v in vertices
        is_duplicate = false
        for existing in cleaned
            if norm(v - existing) < tol
                is_duplicate = true
                break
            end
        end
        if !is_duplicate
            push!(cleaned, v)
        end
    end
    
    # Ensure we have at least 3 vertices for a valid polygon
    if length(cleaned) < 3
        return SVector{2, Float64}[]
    end
    
    # Ensure counter-clockwise ordering
    # Compute centroid
    centroid = sum(cleaned) / length(cleaned)
    
    # Sort by angle from centroid
    angles = [atan(v[2] - centroid[2], v[1] - centroid[1]) for v in cleaned]
    perm = sortperm(angles)
    
    return cleaned[perm]
end

"""
Split cells that are crossed by constraints into multiple cells
Each cell that intersects a constraint is split into separate cells on each side
"""
function _split_cells_by_constraints(cells_data, pts, constraint_edges)
    if isempty(constraint_edges)
        return cells_data
    end
    
    split_cells = []
    
    for cell in cells_data
        # Try to split this cell by all constraints
        # Start with the original cell as a list of candidate cells to split
        current_cells = [(vertices=cell.vertices, center_idx=cell.center_idx)]
        
        # Apply each constraint, potentially splitting cells multiple times
        for (idx1, idx2) in constraint_edges
            p1 = pts[idx1]
            p2 = pts[idx2]
            
            next_cells = []
            for sub_cell in current_cells
                # Try to split this sub-cell by this constraint
                split_result = _split_polygon_by_line_2d(sub_cell.vertices, p1, p2)
                
                if length(split_result) == 1
                    # Cell not split by this constraint - keep as is
                    push!(next_cells, sub_cell)
                else
                    # Cell was split - add both parts with computed centroids
                    for poly_verts in split_result
                        if !isempty(poly_verts)
                            # Compute centroid for this split piece
                            centroid = sum(poly_verts) / length(poly_verts)
                            push!(next_cells, (vertices=poly_verts, center_idx=cell.center_idx, centroid=centroid))
                        end
                    end
                end
            end
            current_cells = next_cells
        end
        
        # Add all resulting cells (original or split versions)
        for final_cell in current_cells
            if !isempty(final_cell.vertices)
                push!(split_cells, final_cell)
            end
        end
    end
    
    return split_cells
end

"""
Split a polygon by a line into two parts (one on each side of the line)
Returns a vector of polygons (1 if no split, 2 if split occurs)
"""
function _split_polygon_by_line_2d(vertices, line_p1, line_p2)
    if isempty(vertices) || length(vertices) < 3
        return [vertices]
    end
    
    # Define the constraint line using point and normal
    line_dir = line_p2 - line_p1
    normal = SVector{2, Float64}(-line_dir[2], line_dir[1])  # Perpendicular to line
    
    tol = 1e-10
    
    # Check if polygon intersects the line
    n = length(vertices)
    distances = [dot(v - line_p1, normal) for v in vertices]
    
    # Count vertices on each side
    n_positive = count(d > tol for d in distances)
    n_negative = count(d < -tol for d in distances)
    
    # If all vertices on one side, no split needed
    if n_positive == 0 || n_negative == 0
        return [vertices]
    end
    
    # Polygon crosses the line - need to split
    # Use Sutherland-Hodgman to clip to each side
    left_poly = SVector{2, Float64}[]
    right_poly = SVector{2, Float64}[]
    
    for i in 1:n
        v1 = vertices[i]
        v2 = vertices[mod1(i + 1, n)]
        
        d1 = distances[i]
        d2 = distances[mod1(i + 1, n)]
        
        # Add v1 to appropriate polygon(s)
        if d1 >= -tol  # On or to the left (positive side)
            push!(left_poly, v1)
        end
        if d1 <= tol  # On or to the right (negative side)
            push!(right_poly, v1)
        end
        
        # If edge crosses the line, add intersection to both polygons
        if (d1 > tol && d2 < -tol) || (d1 < -tol && d2 > tol)
            # Compute intersection
            denom = dot(v2 - v1, normal)
            if abs(denom) > 1e-14
                t = dot(line_p1 - v1, normal) / denom
                t = clamp(t, 0.0, 1.0)
                intersection = v1 + t * (v2 - v1)
                push!(left_poly, intersection)
                push!(right_poly, intersection)
            end
        end
    end
    
    # Clean up the polygons
    left_poly = _clean_polygon_vertices(left_poly)
    right_poly = _clean_polygon_vertices(right_poly)
    
    result = []
    if !isempty(left_poly)
        push!(result, left_poly)
    end
    if !isempty(right_poly)
        push!(result, right_poly)
    end
    
    return result
end

"""
Clip a polygon by a perpendicular bisector
"""
function _clip_polygon_by_bisector_2d(vertices, p1, p2)
    if isempty(vertices)
        return vertices
    end
    
    # Perpendicular bisector of p1 and p2
    # Keep points on the p1 side
    mid = (p1 + p2) / 2
    normal = p2 - p1
    
    clipped = SVector{2, Float64}[]
    n = length(vertices)
    
    for i in 1:n
        v1 = vertices[i]
        v2 = vertices[mod1(i + 1, n)]
        
        # Check which side of the bisector each vertex is on
        d1 = dot(v1 - mid, normal)
        d2 = dot(v2 - mid, normal)
        
        if d1 <= 0
            push!(clipped, v1)
        end
        
        # If edge crosses the bisector, add intersection point
        if (d1 < 0 && d2 > 0) || (d1 > 0 && d2 < 0)
            # Compute intersection
            t = d1 / (d1 - d2)
            intersection = v1 + t * (v2 - v1)
            push!(clipped, intersection)
        end
    end
    
    return clipped
end

"""
Clip a polygon by a constraint line
The polygon is clipped to keep only the part on the same side as the center point
"""
function _clip_polygon_by_line_2d(vertices, line_p1, line_p2, center)
    if isempty(vertices)
        return vertices
    end
    
    # Define the constraint line using point and normal
    # Normal points to the left of the line direction (p1 -> p2)
    line_dir = line_p2 - line_p1
    normal = SVector{2, Float64}(-line_dir[2], line_dir[1])  # Perpendicular to line
    
    # Determine which side of the line the center is on
    center_dist = dot(center - line_p1, normal)
    
    # If center is very close to or on the constraint line, don't clip
    # These are the constraint points themselves
    tol = 1e-10
    if abs(center_dist) < tol
        return vertices
    end
    
    # Clip polygon using Sutherland-Hodgman algorithm
    clipped = SVector{2, Float64}[]
    n = length(vertices)
    
    for i in 1:n
        v1 = vertices[i]
        v2 = vertices[mod1(i + 1, n)]
        
        # Distance from line (positive on same side as normal, negative on other side)
        d1 = dot(v1 - line_p1, normal)
        d2 = dot(v2 - line_p1, normal)
        
        # Check if vertices are on the same side as center
        # Allow vertices to be exactly on the line (within tolerance)
        v1_inside = (d1 * center_dist >= -tol * abs(center_dist))
        v2_inside = (d2 * center_dist >= -tol * abs(center_dist))
        
        if v1_inside
            push!(clipped, v1)
        end
        
        # If edge crosses the constraint line, add intersection point
        if v1_inside != v2_inside
            # Compute intersection with line
            # Parametric line: v1 + t*(v2 - v1)
            # Line equation: dot(point - line_p1, normal) = 0
            # Solve: dot(v1 + t*(v2-v1) - line_p1, normal) = 0
            denom = dot(v2 - v1, normal)
            if abs(denom) > 1e-14
                t = dot(line_p1 - v1, normal) / denom
                t = clamp(t, 0.0, 1.0)  # Ensure t is in [0, 1]
                intersection = v1 + t * (v2 - v1)
                push!(clipped, intersection)
            end
        end
    end
    
    return clipped
end

"""
Build UnstructuredMesh from Voronoi cell data
"""
function _build_unstructured_mesh_2d(cells_data, all_pts)
    # Collect all unique vertices (nodes)
    all_vertices = SVector{2, Float64}[]
    vertex_map = Dict{SVector{2, Float64}, Int}()
    
    function get_or_add_vertex(v)
        # Find existing vertex within tolerance
        for (existing_v, idx) in vertex_map
            if norm(v - existing_v) < 1e-10
                return idx
            end
        end
        push!(all_vertices, v)
        idx = length(all_vertices)
        vertex_map[v] = idx
        return idx
    end
    
    # Build cells and faces
    nc = length(cells_data)
    cells_to_faces_map = [Int[] for _ in 1:nc]
    boundary_cells_to_faces_map = [Int[] for _ in 1:nc]
    
    # Track all edges (potential faces)
    edge_to_face = Dict{Tuple{Int, Int}, Int}()
    faces_nodes_data = Vector{Vector{Int}}()
    face_neighbors_data = Vector{Tuple{Int, Int}}()
    
    boundary_faces_nodes_data = Vector{Vector{Int}}()
    boundary_neighbors_data = Vector{Int}()
    
    for (cell_idx, cell) in enumerate(cells_data)
        vertices = cell.vertices
        nv = length(vertices)
        
        for i in 1:nv
            v1_idx = get_or_add_vertex(vertices[i])
            v2_idx = get_or_add_vertex(vertices[mod1(i + 1, nv)])
            
            # Create edge key (sorted for consistency)
            edge_key = v1_idx < v2_idx ? (v1_idx, v2_idx) : (v2_idx, v1_idx)
            
            if haskey(edge_to_face, edge_key)
                # Internal face - already seen from another cell
                face_idx = edge_to_face[edge_key]
                push!(cells_to_faces_map[cell_idx], face_idx)
                # Update neighbor
                old_neighbor = face_neighbors_data[face_idx]
                face_neighbors_data[face_idx] = (old_neighbor[1], cell_idx)
            else
                # First time seeing this edge - could be internal or boundary
                # For now, assume it's internal; we'll fix boundary later
                face_idx = length(faces_nodes_data) + 1
                edge_to_face[edge_key] = face_idx
                push!(faces_nodes_data, [v1_idx, v2_idx])
                push!(face_neighbors_data, (cell_idx, 0))  # 0 means unknown neighbor
                push!(cells_to_faces_map[cell_idx], face_idx)
            end
        end
    end
    
    # Separate internal and boundary faces
    internal_faces_nodes = Vector{Int}[]
    internal_faces_nodespos = [1]
    internal_neighbors = Tuple{Int, Int}[]
    internal_face_map = Dict{Int, Int}()
    
    boundary_faces_nodes = Vector{Int}[]
    boundary_faces_nodespos = [1]
    boundary_neighbors = Int[]
    boundary_face_map = Dict{Int, Int}()
    
    for (face_idx, (nodes, neighbors)) in enumerate(zip(faces_nodes_data, face_neighbors_data))
        if neighbors[2] == 0
            # Boundary face
            bf_idx = length(boundary_faces_nodes) + 1
            boundary_face_map[face_idx] = bf_idx
            push!(boundary_faces_nodes, nodes)
            push!(boundary_faces_nodespos, boundary_faces_nodespos[end] + length(nodes))
            push!(boundary_neighbors, neighbors[1])
        else
            # Internal face
            if_idx = length(internal_faces_nodes) + 1
            internal_face_map[face_idx] = if_idx
            push!(internal_faces_nodes, nodes)
            push!(internal_faces_nodespos, internal_faces_nodespos[end] + length(nodes))
            push!(internal_neighbors, neighbors)
        end
    end
    
    # Update cells_to_faces to use new face indices
    cells_to_internal = [Int[] for _ in 1:nc]
    cells_to_boundary = [Int[] for _ in 1:nc]
    
    for cell_idx in 1:nc
        for face_idx in cells_to_faces_map[cell_idx]
            if haskey(internal_face_map, face_idx)
                push!(cells_to_internal[cell_idx], internal_face_map[face_idx])
            else
                push!(cells_to_boundary[cell_idx], boundary_face_map[face_idx])
            end
        end
    end
    
    # Flatten cell-to-face mappings
    cells_faces = Int[]
    cells_facepos = [1]
    for faces in cells_to_internal
        append!(cells_faces, faces)
        push!(cells_facepos, cells_facepos[end] + length(faces))
    end
    
    boundary_cells_faces = Int[]
    boundary_cells_facepos = [1]
    for faces in cells_to_boundary
        append!(boundary_cells_faces, faces)
        push!(boundary_cells_facepos, boundary_cells_facepos[end] + length(faces))
    end
    
    # Flatten faces-to-nodes
    faces_nodes_flat = if isempty(internal_faces_nodes)
        Int[]
    else
        reduce(vcat, internal_faces_nodes)
    end
    boundary_faces_nodes_flat = if isempty(boundary_faces_nodes)
        Int[]
    else
        reduce(vcat, boundary_faces_nodes)
    end
    
    # Create UnstructuredMesh
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
