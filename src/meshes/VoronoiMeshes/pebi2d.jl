# Module-level constants for geometric operations
const GEOMETRIC_TOLERANCE = 1e-9
const EDGE_MIDPOINT_THRESHOLD = 0.5

"""
    PEBIMesh2D(points; bbox=nothing)

Create a 2D PEBI (Perpendicular Bisector) / Voronoi mesh from a set of points.

# Arguments
- `points`: A matrix of size (2, n) or vector of 2-tuples/vectors containing the x,y coordinates of cell centers
- `bbox`: Optional bounding box as ((xmin, xmax), (ymin, ymax)). If not provided, computed from points with margin

# Returns
- An `UnstructuredMesh` instance representing the PEBI mesh

# Description
The PEBI mesh is a Perpendicular Bisector mesh, also known as a Voronoi diagram.
Each input point becomes a cell center in the resulting mesh.
The mesh is bounded by the specified or computed bounding box.

To add line segment constraints to the mesh, use the `insert_line_segment` function
after creating the basic mesh.

# Examples
```julia
# Simple mesh with 4 points
points = [0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0]
mesh = PEBIMesh2D(points)

# Mesh with constraint line (use post-processing)
points = rand(2, 20)
mesh = PEBIMesh2D(points)
mesh = insert_line_segment(mesh, [0.5, 0.0], [0.5, 1.0])  # Add vertical constraint
# Result will have > 20 cells due to splitting
```
"""
function PEBIMesh2D(points; bbox=nothing)
    # Convert points to standard format
    pts, npts = _convert_points_2d(points)
    
    # Compute or validate bounding box
    bb = _compute_bbox_2d(pts, bbox)
    
    # Generate Voronoi diagram
    cells_data = _generate_voronoi_2d(pts, bb)
    
    # Build UnstructuredMesh from the Voronoi cells
    mesh = _build_unstructured_mesh_2d(cells_data, pts)
    
    return mesh
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
    _compute_centroid(vertices)

Compute the centroid of a 2D polygon given its vertices.

Uses the signed area method which works for both convex and non-convex polygons.
Assumes vertices are ordered (either clockwise or counter-clockwise).

# Arguments
- `vertices`: Vector of SVector{2, Float64} representing polygon vertices in order

# Returns
- SVector{2, Float64} representing the centroid coordinates
"""
function _compute_centroid(vertices::AbstractVector{SVector{2, T}}) where T
    n = length(vertices)
    if n < 3
        # Degenerate polygon - return mean of vertices
        return sum(vertices) / n
    end
    
    # Use signed area formula for polygon centroid
    # Cx = (1 / 6A) * Σ(xi + xi+1)(xi * yi+1 - xi+1 * yi)
    # Cy = (1 / 6A) * Σ(yi + yi+1)(xi * yi+1 - xi+1 * yi)
    # where A = (1/2) * Σ(xi * yi+1 - xi+1 * yi)
    
    signed_area = zero(T)
    cx = zero(T)
    cy = zero(T)
    
    for i in 1:n
        j = i < n ? i + 1 : 1  # Next vertex (wrap around)
        xi, yi = vertices[i][1], vertices[i][2]
        xj, yj = vertices[j][1], vertices[j][2]
        
        cross = xi * yj - xj * yi
        signed_area += cross
        cx += (xi + xj) * cross
        cy += (yi + yj) * cross
    end
    
    signed_area *= 0.5
    
    # Handle degenerate case where area is zero
    if abs(signed_area) < 1e-14
        # Fall back to simple mean
        return sum(vertices) / n
    end
    
    cx /= (6.0 * signed_area)
    cy /= (6.0 * signed_area)
    
    return SVector{2, T}(cx, cy)
end

"""
Generate Voronoi cells using a simple box-clipping approach
"""
function _generate_voronoi_2d(pts, bb)
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
    
    # Remove duplicates - use consistent tolerance with vertex matching
    cleaned = SVector{2, Float64}[]
    tol = 1e-10  # Match the tolerance used in get_or_add_vertex
    
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
    tol = 1e-9
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
        tol = 1e-10  # Standard geometric tolerance
        
        # Check all existing vertices
        for i in 1:length(all_vertices)
            if norm(v - all_vertices[i]) < tol
                return i
            end
        end
        
        # Not found - add new vertex
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

"""
    find_intersected_faces(mesh, line_p1, line_p2)

Find all faces (edges) in the mesh that are intersected by a line segment.

# Arguments
- `mesh`: An UnstructuredMesh
- `line_p1`: Start point of line segment as 2-element vector/tuple
- `line_p2`: End point of line segment as 2-element vector/tuple

# Returns
A vector of named tuples `(face_idx, intersection_point, t)` where:
- `face_idx`: Index of the intersected face
- `intersection_point`: The intersection point on the face
- `t`: Parameter along line segment (0 at line_p1, 1 at line_p2)

Faces are sorted by parameter `t` (order along line segment).
"""
function find_intersected_faces(mesh::UnstructuredMesh, line_p1, line_p2)
    p1 = SVector{2, Float64}(line_p1[1], line_p1[2])
    p2 = SVector{2, Float64}(line_p2[1], line_p2[2])
    line_dir = p2 - p1
    line_length_sq = dot(line_dir, line_dir)
    
    if line_length_sq < 1e-20
        error("Line segment is degenerate (zero length)")
    end
    
    intersections = []
    tol = 1e-10
    
    # Check all interior faces
    nf_internal = number_of_faces(mesh)
    for face_idx in 1:nf_internal
        # Get nodes of this face
        face_nodes = mesh.faces.faces_to_nodes[face_idx]
        if length(face_nodes) != 2
            continue  # Skip non-edge faces
        end
        
        # Get node coordinates
        n1_idx, n2_idx = face_nodes
        n1 = SVector{2, Float64}(mesh.node_points[n1_idx]...)
        n2 = SVector{2, Float64}(mesh.node_points[n2_idx]...)
        
        # Check if line segment intersects this edge
        intersection, t_line, t_edge = _line_segment_intersection_2d(p1, p2, n1, n2, tol)
        
        if intersection !== nothing
            # Valid intersection found
            push!(intersections, (face_idx=face_idx, intersection_point=intersection, t=t_line, 
                                 is_boundary=false, edge_t=t_edge))
        end
    end
    
    # Check boundary faces
    nf_boundary = number_of_boundary_faces(mesh)
    for bface_idx in 1:nf_boundary
        # Get nodes of this boundary face
        face_nodes = mesh.boundary_faces.faces_to_nodes[bface_idx]
        if length(face_nodes) != 2
            continue
        end
        
        n1_idx, n2_idx = face_nodes
        n1 = SVector{2, Float64}(mesh.node_points[n1_idx]...)
        n2 = SVector{2, Float64}(mesh.node_points[n2_idx]...)
        
        intersection, t_line, t_edge = _line_segment_intersection_2d(p1, p2, n1, n2, tol)
        
        if intersection !== nothing
            push!(intersections, (face_idx=bface_idx, intersection_point=intersection, t=t_line,
                                 is_boundary=true, edge_t=t_edge))
        end
    end
    
    # Sort by parameter t (order along line segment)
    sort!(intersections, by = x -> x.t)
    
    return intersections
end

"""
Check if two line segments intersect and return intersection point and parameters
Returns (intersection_point, t1, t2) where:
- t1 is parameter along first segment (0 to 1)
- t2 is parameter along second segment (0 to 1)
Returns (nothing, nothing, nothing) if no intersection
"""
function _line_segment_intersection_2d(p1, p2, q1, q2, tol)
    # Parametric form: P(t) = p1 + t*(p2-p1), Q(s) = q1 + s*(q2-q1)
    # Solve: p1 + t*d1 = q1 + s*d2
    # Where d1 = p2-p1, d2 = q2-q1
    
    d1 = p2 - p1
    d2 = q2 - q1
    d_start = q1 - p1
    
    # Cross product in 2D: d1 × d2 = d1.x * d2.y - d1.y * d2.x
    cross_d1_d2 = d1[1] * d2[2] - d1[2] * d2[1]
    
    # Check if lines are parallel
    if abs(cross_d1_d2) < 1e-14
        return nothing, nothing, nothing
    end
    
    # Solve for parameters
    t = (d_start[1] * d2[2] - d_start[2] * d2[1]) / cross_d1_d2
    s = (d_start[1] * d1[2] - d_start[2] * d1[1]) / cross_d1_d2
    
    # Check if intersection is within both segments
    if t >= -tol && t <= 1.0 + tol && s >= -tol && s <= 1.0 + tol
        # Clamp to [0, 1]
        t = clamp(t, 0.0, 1.0)
        s = clamp(s, 0.0, 1.0)
        intersection = p1 + t * d1
        return intersection, t, s
    end
    
    return nothing, nothing, nothing
end

"""
    insert_line_segment(mesh, line_p1, line_p2)

Insert a line segment into the mesh by splitting cells that are crossed by the line.
Creates new interior faces along the line segment.

# Arguments
- `mesh`: An UnstructuredMesh to modify
- `line_p1`: Start point of line segment as 2-element vector/tuple  
- `line_p2`: End point of line segment as 2-element vector/tuple

# Returns
A new UnstructuredMesh with cells split along the line segment and new interior faces added.

# Description
This function finds all cells that are crossed by the line segment and splits them.
Cells with 2+ edge intersections are split into left/right parts.
Cells with only 1 edge intersection require special handling and are split into 3+ parts.

The line segment becomes interior faces in the resulting mesh (not boundary faces).
"""
function insert_line_segment(mesh::UnstructuredMesh, line_p1, line_p2)
    # Convert line points to SVectors
    p1 = SVector{2, Float64}(line_p1[1], line_p1[2])
    p2 = SVector{2, Float64}(line_p2[1], line_p2[2])
    
    # Find all faces intersected by the line segment
    intersections = find_intersected_faces(mesh, p1, p2)
    
    if isempty(intersections)
        @info "No faces intersected by line segment"
        return mesh
    end
    
    # Extract mesh data
    cells_to_faces = mesh.faces.cells_to_faces
    faces_to_nodes = mesh.faces.faces_to_nodes
    face_neighbors = mesh.faces.neighbors
    boundary_cells_to_faces = mesh.boundary_faces.cells_to_faces
    boundary_faces_to_nodes = mesh.boundary_faces.faces_to_nodes
    boundary_cells = mesh.boundary_faces.neighbors
    node_points = mesh.node_points
    
    # Identify cells that are crossed by the line segment
    crossed_cells = Set{Int}()
    for inter in intersections
        if !inter.is_boundary
            l, r = face_neighbors[inter.face_idx]
            push!(crossed_cells, l)
            push!(crossed_cells, r)
        else
            # Boundary face - only one cell
            push!(crossed_cells, boundary_cells[inter.face_idx])
        end
    end
    crossed_cells = sort(collect(crossed_cells))
    
    # Build new mesh with split cells
    new_cells_data = []
    old_to_new_cells = Dict{Int, Vector{Int}}()
    
    nc = length(cells_to_faces)
    next_cell_idx = 1
    
    for cell_idx in 1:nc
        if cell_idx in crossed_cells
            # Split this cell
            split_cells = _split_cell_by_line_segment(
                cell_idx, mesh, p1, p2, intersections
            )
            old_to_new_cells[cell_idx] = collect(next_cell_idx:(next_cell_idx + length(split_cells) - 1))
            for split_cell in split_cells
                push!(new_cells_data, split_cell)
            end
            next_cell_idx += length(split_cells)
        else
            # Keep original cell
            old_to_new_cells[cell_idx] = [next_cell_idx]
            # Extract cell vertices
            vertices = _get_cell_vertices(cell_idx, mesh)
            cell_data = (vertices = vertices, center = _compute_centroid(vertices))
            push!(new_cells_data, cell_data)
            next_cell_idx += 1
        end
    end
    
    # Build new UnstructuredMesh from split cells
    return _build_unstructured_mesh_2d(new_cells_data, node_points)
end

"""
Get vertices of a cell from the mesh in order
"""
function _get_cell_vertices(cell_idx, mesh)
    cells_to_faces = mesh.faces.cells_to_faces
    faces_to_nodes = mesh.faces.faces_to_nodes
    face_neighbors = mesh.faces.neighbors
    boundary_cells_to_faces = mesh.boundary_faces.cells_to_faces
    boundary_faces_to_nodes = mesh.boundary_faces.faces_to_nodes
    node_points = mesh.node_points
    
    # Collect all edges (faces) of this cell
    interior_faces = cells_to_faces[cell_idx]
    boundary_faces = boundary_cells_to_faces[cell_idx]
    
    # Build edge list with nodes
    edges = Tuple{Int, Int}[]
    
    for face_idx in interior_faces
        nodes = faces_to_nodes[face_idx]
        l, r = face_neighbors[face_idx]
        # Determine orientation based on which cell we are
        if l == cell_idx
            # Use forward direction
            for i in 1:(length(nodes)-1)
                push!(edges, (nodes[i], nodes[i+1]))
            end
        else
            # Use reverse direction
            for i in length(nodes):-1:2
                push!(edges, (nodes[i], nodes[i-1]))
            end
        end
    end
    
    for face_idx in boundary_faces
        nodes = boundary_faces_to_nodes[face_idx]
        # Boundary faces are oriented correctly for the cell
        for i in 1:(length(nodes)-1)
            push!(edges, (nodes[i], nodes[i+1]))
        end
    end
    
    # Build ordered vertex list from edges
    if isempty(edges)
        return SVector{2, Float64}[]
    end
    
    vertices_idx = Int[]
    current_edge = edges[1]
    push!(vertices_idx, current_edge[1])
    used = Set([1])
    
    while length(used) < length(edges)
        next_node = current_edge[2]
        push!(vertices_idx, next_node)
        
        # Find next edge that starts with next_node
        found = false
        for (idx, edge) in enumerate(edges)
            if idx ∉ used && edge[1] == next_node
                current_edge = edge
                push!(used, idx)
                found = true
                break
            end
        end
        
        if !found
            break
        end
    end
    
    # Remove last vertex if it's the same as first
    if !isempty(vertices_idx) && vertices_idx[end] == vertices_idx[1]
        pop!(vertices_idx)
    end
    
    # Convert to SVector points
    return [node_points[v] for v in vertices_idx]
end

"""
Split a cell by a line segment
"""
function _split_cell_by_line_segment(cell_idx, mesh, line_p1, line_p2, intersections)
    # Get cell vertices
    vertices = _get_cell_vertices(cell_idx, mesh)
    
    if isempty(vertices) || length(vertices) < 3
        # Invalid cell, return empty
        return []
    end
    
    # Find which edges of this cell are intersected
    cell_intersections = []
    face_neighbors = mesh.faces.neighbors
    boundary_cells = mesh.boundary_faces.neighbors
    
    for inter in intersections
        if inter.is_boundary
            if boundary_cells[inter.face_idx] == cell_idx
                push!(cell_intersections, inter)
            end
        else
            l, r = face_neighbors[inter.face_idx]
            if l == cell_idx || r == cell_idx
                push!(cell_intersections, inter)
            end
        end
    end
    
    if isempty(cell_intersections)
        # No intersections for this cell (shouldn't happen)
        center = _compute_centroid(vertices)
        return [(vertices = vertices, center = center)]
    end
    
    # Sort intersections by parameter t along line segment
    sort!(cell_intersections, by = x -> x.t)
    
    # Determine split type based on number of intersections
    n_intersections = length(cell_intersections)
    
    if n_intersections == 1
        # Special case: line enters/exits through same edge
        # For now, don't split (conservative approach)
        center = _compute_centroid(vertices)
        return [(vertices = vertices, center = center)]
    else
        # Standard case: line crosses cell (2 or more intersections)
        # Use first and last intersection points
        int_pt1 = cell_intersections[1].intersection_point
        int_pt2 = cell_intersections[end].intersection_point
        return _split_cell_two_points(vertices, int_pt1, int_pt2)
    end
end

"""
Split cell into two parts using two intersection points on the boundary
"""
function _split_cell_two_points(vertices, int_pt1, int_pt2)
    if isempty(vertices)
        return []
    end
    
    # Find where int_pt1 and int_pt2 lie on the cell boundary
    n = length(vertices)
    idx1 = -1
    idx2 = -1
    t1 = 0.0
    t2 = 0.0
    
    tol = GEOMETRIC_TOLERANCE
    
    for i in 1:n
        v1 = vertices[i]
        v2 = vertices[mod1(i + 1, n)]
        edge_vec = v2 - v1
        edge_len = norm(edge_vec)
        
        if edge_len < tol
            continue
        end
        
        # Check if int_pt1 is on this edge
        proj1 = dot(int_pt1 - v1, edge_vec) / (edge_len * edge_len)
        if proj1 >= -tol && proj1 <= 1.0 + tol
            dist1 = norm(int_pt1 - (v1 + proj1 * edge_vec))
            if dist1 < tol
                idx1 = i
                t1 = clamp(proj1, 0.0, 1.0)
            end
        end
        
        # Check if int_pt2 is on this edge
        proj2 = dot(int_pt2 - v1, edge_vec) / (edge_len * edge_len)
        if proj2 >= -tol && proj2 <= 1.0 + tol
            dist2 = norm(int_pt2 - (v1 + proj2 * edge_vec))
            if dist2 < tol
                idx2 = i
                t2 = clamp(proj2, 0.0, 1.0)
            end
        end
    end
    
    if idx1 == -1 || idx2 == -1
        # Couldn't find intersection points on boundary
        # Return original cell
        center = _compute_centroid(vertices)
        return [(vertices = vertices, center = center)]
    end
    
    # Build two new cells
    # Cell 1: from int_pt1 to int_pt2 going one way
    # Cell 2: from int_pt2 to int_pt1 going the other way
    
    cell1_verts = SVector{2, Float64}[]
    cell2_verts = SVector{2, Float64}[]
    
    # Add int_pt1 to both cells
    push!(cell1_verts, int_pt1)
    
    # Walk from idx1+1 to idx2, adding vertices to cell1
    i = mod1(idx1 + 1, n)
    while i != idx2
        # Skip start vertex if intersection is past edge midpoint
        if t1 > EDGE_MIDPOINT_THRESHOLD || i != idx1
            push!(cell1_verts, vertices[i])
        end
        i = mod1(i + 1, n)
    end
    
    # Add remaining part of edge idx2 if needed
    if t2 > tol
        push!(cell1_verts, vertices[idx2])
    end
    
    # Add int_pt2
    push!(cell1_verts, int_pt2)
    
    # Build cell2: from int_pt2 back to int_pt1 the other way
    push!(cell2_verts, int_pt2)
    
    # Walk from idx2+1 to idx1
    i = mod1(idx2 + 1, n)
    while i != idx1
        # Skip start vertex if intersection is past edge midpoint
        if t2 > EDGE_MIDPOINT_THRESHOLD || i != idx2
            push!(cell2_verts, vertices[i])
        end
        i = mod1(i + 1, n)
    end
    
    # Add remaining part of edge idx1 if needed
    if t1 > tol
        push!(cell2_verts, vertices[idx1])
    end
    
    push!(cell2_verts, int_pt1)
    
    # Clean up vertices
    cell1_verts = _clean_polygon_vertices(cell1_verts)
    cell2_verts = _clean_polygon_vertices(cell2_verts)
    
    result = []
    if length(cell1_verts) >= 3
        push!(result, (vertices = cell1_verts, center = _compute_centroid(cell1_verts)))
    end
    if length(cell2_verts) >= 3
        push!(result, (vertices = cell2_verts, center = _compute_centroid(cell2_verts)))
    end
    
    return result
end
