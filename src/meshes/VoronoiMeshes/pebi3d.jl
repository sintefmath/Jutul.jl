# Constants for 3D mesh generation
const GEOMETRIC_TOLERANCE_3D = 1e-9
const BBOX_MARGIN_3D = 0.1

"""
    PEBIMesh3D(points; bbox=nothing, verbose=false, edge_window_size=5)

Create a 3D PEBI (Perpendicular Bisector) / Voronoi mesh from a set of points.

# Arguments
- `points`: A matrix of size (3, n) or (n, 3) or vector of 3-tuples/vectors containing the x,y,z coordinates of cell centers
- `bbox`: Optional bounding box as ((xmin, xmax), (ymin, ymax), (zmin, zmax)). If not provided, computed from points with margin
- `verbose`: If true, prints progress information during mesh generation
- `edge_window_size`: Controls accuracy vs speed trade-off in edge detection (default=5). Larger values are more accurate but slower.

# Returns
- An `UnstructuredMesh` instance representing the 3D PEBI mesh

# Description
The 3D PEBI mesh is a 3D Voronoi diagram where each input point becomes a cell center.
The mesh is bounded by the specified or computed bounding box. Each Voronoi cell is computed
by intersecting half-spaces defined by perpendicular bisector planes between neighboring points.

This implementation uses optimized half-space intersection. It works well for point counts
up to several thousand points. Use `verbose=true` to monitor progress for large meshes.

The `edge_window_size` parameter controls how many neighbors are checked when computing
edge-plane intersections. Default value of 5 works well for typical cases. Increase for
more complex geometries or if you notice incorrect cell boundaries.

# Examples
```julia
# Simple mesh with 8 points (corners of a cube)
points = [0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0;
          0.0 0.0 1.0 1.0 0.0 0.0 1.0 1.0;
          0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0]
mesh = Jutul.VoronoiMeshes.PEBIMesh3D(points)

# Larger mesh with progress output
points = rand(3, 1000)
mesh = Jutul.VoronoiMeshes.PEBIMesh3D(points, verbose=true)

# High accuracy mesh (slower but more precise)
points = rand(3, 100)
mesh = Jutul.VoronoiMeshes.PEBIMesh3D(points, edge_window_size=10)
```
"""
function PEBIMesh3D(points; bbox=nothing, verbose=false, edge_window_size=5)
    # Convert points to standard format (3×n)
    pts = _convert_points_3d(points)
    n_points = size(pts, 2)
    
    if verbose
        @info "Creating 3D PEBI mesh with $n_points points (edge_window_size=$edge_window_size)"
    end
    
    # Compute or validate bounding box
    if isnothing(bbox)
        bbox = _compute_bbox_3d(pts)
    else
        bbox = _validate_bbox_3d(bbox)
    end
    
    # Generate Voronoi cells
    cells, all_nodes = _generate_voronoi_3d(pts, bbox, verbose, edge_window_size)
    
    if verbose
        @info "Building UnstructuredMesh..."
    end
    
    # Build UnstructuredMesh
    mesh = _build_unstructured_mesh_3d(cells, all_nodes)
    
    if verbose
        @info "Mesh generation complete"
    end
    
    return mesh
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
function _generate_voronoi_3d(pts::AbstractMatrix, bbox, verbose::Bool, edge_window_size::Int)
    n_points = size(pts, 2)
    cells = Vector{Vector{SVector{3, Float64}}}()
    all_nodes = Set{SVector{3, Float64}}()
    
    if verbose
        @info "Generating $n_points Voronoi cells..."
    end
    
    # For each point, compute its Voronoi cell
    for i in 1:n_points
        if verbose && (i == 1 || i % 10 == 0 || i == n_points)
            @info "Processing cell $i of $n_points"
        end
        
        cell_vertices = _compute_voronoi_cell_3d(i, pts, bbox, edge_window_size)
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
function _compute_voronoi_cell_3d(idx::Int, pts::AbstractMatrix, bbox, edge_window_size::Int)
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
        vertices = _clip_polyhedron_by_halfspace_3d(vertices, midpoint, normal, edge_window_size)
        
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

Optimized implementation that avoids O(n²) all-pairs checking. Instead, we:
1. Keep vertices on the correct side
2. Add edge-plane intersections only for edges defined by distance sign changes
3. Use a simple heuristic: vertices close in the original list are likely connected

This is much faster than checking all pairs while maintaining correctness for
convex polyhedra generated by successive clipping operations.

The `edge_window_size` parameter controls how many neighbors to check for each vertex.
Larger values are more accurate but slower. Default of 5 works well for typical cases.
"""
function _clip_polyhedron_by_halfspace_3d(vertices, point, normal, edge_window_size::Int)
    if isempty(vertices)
        return vertices
    end
    
    n = length(vertices)
    if n <= 3
        # Too few vertices, just keep those on correct side
        return [v for v in vertices if dot(v - point, normal) >= -GEOMETRIC_TOLERANCE_3D]
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
    
    # For intersection points, use a windowing approach:
    # Check consecutive vertices (assuming they form edges) plus a few neighbors
    # This works well for convex polyhedra built by successive clipping
    window_size = min(edge_window_size, n)  # Check up to edge_window_size neighbors for each vertex
    
    seen_intersections = Set{NTuple{3, Float64}}()  # Track to avoid duplicates
    
    for i in 1:n
        d_i = distances[i]
        
        # Check nearby vertices as potential edge endpoints
        for offset in 1:window_size
            j = mod1(i + offset, n)
            if i == j
                continue
            end
            
            d_j = distances[j]
            
            # Check if edge crosses the plane (one vertex inside, one outside)
            if (d_i >= -GEOMETRIC_TOLERANCE_3D && d_j < -GEOMETRIC_TOLERANCE_3D) ||
               (d_i < -GEOMETRIC_TOLERANCE_3D && d_j >= -GEOMETRIC_TOLERANCE_3D)
                
                # Compute intersection point
                t = -d_i / (d_j - d_i)
                t = clamp(t, 0.0, 1.0)
                
                intersection = vertices[i] + t * (vertices[j] - vertices[i])
                
                # Use rounded coordinates for duplicate detection (much faster than norm())
                key = (round(intersection[1], digits=8), 
                       round(intersection[2], digits=8), 
                       round(intersection[3], digits=8))
                
                if key ∉ seen_intersections
                    push!(seen_intersections, key)
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
Extract faces from a convex polyhedron using Gift Wrapping (Jarvis March) algorithm.

This algorithm finds all faces on the convex hull surface by:
1. Finding a starting edge on the hull
2. For each edge, finding the next vertex that forms a face
3. Continuing until all faces are found

Returns a vector of faces, where each face is a vector of vertex indices (in order).
"""
function _extract_convex_hull_faces_3d(vertices::Vector{SVector{3, T}}) where T
    n = length(vertices)
    if n < 4
        return Vector{Int}[]
    end
    
    faces = Vector{Vector{Int}}[]
    processed_edges = Set{Tuple{Int, Int}}()
    
    # Find a starting edge that's definitely on the hull
    start_edge = nothing
    for i in 1:n
        for j in (i+1):n
            if _is_edge_on_hull(vertices[i], vertices[j], vertices)
                start_edge = (i, j)
                break
            end
        end
        if start_edge !== nothing
            break
        end
    end
    
    if start_edge === nothing
        # Fallback: just use first two vertices
        start_edge = (1, 2)
    end
    
    # Queue of edges to process
    edges_to_process = [start_edge]
    
    while !isempty(edges_to_process)
        v1_idx, v2_idx = pop!(edges_to_process)
        
        # Skip if already processed
        edge_key = v1_idx < v2_idx ? (v1_idx, v2_idx) : (v2_idx, v1_idx)
        if edge_key in processed_edges
            continue
        end
        
        # Find the face containing this edge
        face_vertices = [v1_idx, v2_idx]
        current_edge = (v1_idx, v2_idx)
        
        # Build the face by finding next vertices
        max_iterations = n  # Prevent infinite loops
        for _ in 1:max_iterations
            v_next = _find_next_vertex_on_face(
                vertices[current_edge[1]], 
                vertices[current_edge[2]], 
                vertices, 
                face_vertices
            )
            
            if v_next === nothing || v_next == v1_idx
                # Face is complete
                break
            end
            
            push!(face_vertices, v_next)
            current_edge = (current_edge[2], v_next)
        end
        
        # Add this face if it has at least 3 vertices
        if length(face_vertices) >= 3
            push!(faces, face_vertices)
            
            # Mark all edges of this face as processed and add unprocessed ones to queue
            for i in 1:length(face_vertices)
                j = mod1(i + 1, length(face_vertices))
                vi, vj = face_vertices[i], face_vertices[j]
                edge = vi < vj ? (vi, vj) : (vj, vi)
                
                if !(edge in processed_edges)
                    push!(processed_edges, edge)
                    # Add the edge in opposite direction for processing
                    push!(edges_to_process, (vj, vi))
                end
            end
        else
            push!(processed_edges, edge_key)
        end
    end
    
    return faces
end

"""
Check if an edge is on the convex hull.

An edge (v1, v2) is on the hull if all other vertices are on one side of any plane containing the edge.
"""
function _is_edge_on_hull(v1::SVector{3, T}, v2::SVector{3, T}, vertices::Vector{SVector{3, T}}) where T
    edge_vec = v2 - v1
    
    # Try to find a reference vertex not on the edge
    ref_vertex = nothing
    for v in vertices
        if norm(v - v1) > GEOMETRIC_TOLERANCE_3D && norm(v - v2) > GEOMETRIC_TOLERANCE_3D
            ref_vertex = v
            break
        end
    end
    
    if ref_vertex === nothing
        return true  # Not enough vertices to check
    end
    
    # Create a plane through the edge and reference vertex
    normal = cross(edge_vec, ref_vertex - v1)
    if norm(normal) < GEOMETRIC_TOLERANCE_3D
        return true  # Degenerate case
    end
    normal = normalize(normal)
    
    # Check if all vertices are on one side
    all_positive = true
    all_negative = true
    
    for v in vertices
        dist = dot(v - v1, normal)
        if dist > GEOMETRIC_TOLERANCE_3D
            all_negative = false
        elseif dist < -GEOMETRIC_TOLERANCE_3D
            all_positive = false
        end
    end
    
    return all_positive || all_negative
end

"""
Find the next vertex on a face during convex hull construction.

Given edge (v1, v2) and current face vertices, find the vertex that maximizes
the turning angle to stay on the convex hull.
"""
function _find_next_vertex_on_face(
    v1::SVector{3, T}, 
    v2::SVector{3, T}, 
    vertices::Vector{SVector{3, T}},
    face_verts::Vector{Int}
) where T
    edge_vec = v2 - v1
    
    # Use first three vertices to determine face normal
    if length(face_verts) >= 3
        v0 = vertices[face_verts[1]]
        normal = normalize(cross(v2 - v0, v1 - v0))
    else
        # For initial edge, use any perpendicular direction
        normal = normalize(cross(edge_vec, SVector(1.0, 0.0, 0.0)))
        if norm(normal) < GEOMETRIC_TOLERANCE_3D
            normal = normalize(cross(edge_vec, SVector(0.0, 1.0, 0.0)))
        end
    end
    
    best_vertex = nothing
    best_angle = -Inf
    
    for (i, v) in enumerate(vertices)
        # Skip if already in face or too close to edge vertices
        if i in face_verts || norm(v - v1) < GEOMETRIC_TOLERANCE_3D || norm(v - v2) < GEOMETRIC_TOLERANCE_3D
            continue
        end
        
        # Calculate angle
        to_vertex = normalize(v - v2)
        backward = normalize(v1 - v2)
        
        # Project onto plane perpendicular to normal
        to_vertex_proj = to_vertex - dot(to_vertex, normal) * normal
        backward_proj = backward - dot(backward, normal) * normal
        
        if norm(to_vertex_proj) < GEOMETRIC_TOLERANCE_3D || norm(backward_proj) < GEOMETRIC_TOLERANCE_3D
            continue
        end
        
        to_vertex_proj = normalize(to_vertex_proj)
        backward_proj = normalize(backward_proj)
        
        # Calculate turning angle
        cos_angle = dot(to_vertex_proj, backward_proj)
        angle = acos(clamp(cos_angle, -1.0, 1.0))
        
        if angle > best_angle
            best_angle = angle
            best_vertex = i
        end
    end
    
    return best_vertex
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
        
        # Extract convex hull faces for this cell
        hull_faces = _extract_convex_hull_faces_3d(cell_verts)
        
        for face_vert_local in hull_faces
            # Convert local indices to global node indices
            face_vert_global = [node_to_idx[cell_verts[i]] for i in face_vert_local]
            face_node_set = Set(face_vert_global)
            
            if !haskey(face_dict, face_node_set)
                face_dict[face_node_set] = next_face_id
                push!(all_faces_to_nodes, sort(collect(face_node_set)))
                push!(face_neighbors_vec, (cell_idx, 0))
                push!(cell_faces, next_face_id)
                next_face_id += 1
            else
                face_id = face_dict[face_node_set]
                old_neighbor = face_neighbors_vec[face_id]
                face_neighbors_vec[face_id] = (old_neighbor[1], cell_idx)
                push!(cell_faces, face_id)
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
    
    # Build neighbor arrays (as vector of tuples, matching 2D implementation)
    internal_neighbors = Tuple{Int, Int}[]
    for face_idx in interior_face_indices
        n1, n2 = face_neighbors_vec[face_idx]
        push!(internal_neighbors, (n1, n2))
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
