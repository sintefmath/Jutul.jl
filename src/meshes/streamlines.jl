# Streamline tracing functionality for UnstructuredMesh
# This module provides functionality for tracing streamlines through unstructured meshes
# by subdividing cells into centroid tesselations and using an octree for fast queries.

export StreamlineTracer, setup_streamline_tracer, trace_streamlines

"""
    OctreeNode{D, T}

Node in an octree spatial index structure. Used for efficient point location queries.
"""
mutable struct OctreeNode{D, T}
    "Bounding box minimum coordinates"
    bbox_min::SVector{D, T}
    "Bounding box maximum coordinates"
    bbox_max::SVector{D, T}
    "Child nodes (8 for 3D, 4 for 2D)"
    children::Union{Nothing, Vector{OctreeNode{D, T}}}
    "Indices of sub-cells contained in this node (leaf nodes only)"
    subcell_indices::Vector{Int}
    "Maximum depth of this subtree"
    max_depth::Int
end

"""
    SubCell{D, T}

Represents a tetrahedral/triangular sub-cell created from cell tesselation.
Each sub-cell has vertices, a centroid, and reconstructed velocity.
"""
struct SubCell{D, T}
    "Parent cell index in the mesh"
    parent_cell::Int
    "Vertex coordinates"
    vertices::Vector{SVector{D, T}}
    "Centroid of the sub-cell"
    centroid::SVector{D, T}
    "Reconstructed velocity at the centroid"
    velocity::SVector{D, T}
    "Volume/area of the sub-cell"
    measure::T
end

"""
    StreamlineTracer{D, T}

Container for streamline tracing setup. This structure contains all the preprocessed
data needed for fast streamline tracing from arbitrary starting points.

The setup phase subdivides each cell into tetrahedral (3D) or triangular (2D) sub-cells
based on centroid tesselation, reconstructs velocities in each sub-cell from face fluxes,
and builds an octree spatial index for fast point location.

# Fields
- `mesh`: The unstructured mesh
- `subcells`: Vector of all sub-cells
- `octree`: Root node of the octree spatial index
- `geometry`: TPFV geometry information
"""
struct StreamlineTracer{D, T}
    "The underlying mesh"
    mesh::UnstructuredMesh{D}
    "Sub-cells from tesselation"
    subcells::Vector{SubCell{D, T}}
    "Octree root for spatial queries"
    octree::OctreeNode{D, T}
    "Geometry information"
    geometry::TwoPointFiniteVolumeGeometry
end

"""
    setup_streamline_tracer(mesh::UnstructuredMesh, fluxes::AbstractVector; geometry = tpfv_geometry(mesh), max_depth = 8)

Setup phase for streamline tracing. This function:
1. Subdivides each cell into centroid-based tesselations
2. Reconstructs velocities from face fluxes for each sub-cell
3. Builds an octree spatial index for fast point location

# Arguments
- `mesh`: The unstructured mesh
- `fluxes`: Vector of face fluxes (one value per interior face)
- `geometry`: Optional pre-computed geometry (defaults to tpfv_geometry(mesh))
- `max_depth`: Maximum octree depth for spatial indexing

# Returns
A `StreamlineTracer` object that can be used with `trace_streamlines`.
"""
function setup_streamline_tracer(mesh::UnstructuredMesh{D}, fluxes::AbstractVector; 
                                 geometry = tpfv_geometry(mesh), 
                                 max_depth::Int = 8) where D
    T = float_type(mesh)
    nc = number_of_cells(mesh)
    
    # Create sub-cells from centroid tesselation and reconstruct velocities
    subcells = SubCell{D, T}[]
    
    for cell in 1:nc
        cell_subcells = tesselate_cell_and_reconstruct_velocity(mesh, cell, fluxes, geometry)
        append!(subcells, cell_subcells)
    end
    
    # Build octree for fast spatial queries
    bbox_min, bbox_max = compute_global_bbox(mesh)
    octree = build_octree(subcells, bbox_min, bbox_max, max_depth)
    
    return StreamlineTracer(mesh, subcells, octree, geometry)
end

"""
    tesselate_cell_and_reconstruct_velocity(mesh, cell, fluxes, geometry)

Subdivide a cell into sub-cells using centroid tesselation and reconstruct
velocity in each sub-cell from face fluxes.
"""
function tesselate_cell_and_reconstruct_velocity(mesh::UnstructuredMesh{D}, cell::Int, 
                                                   fluxes::AbstractVector, 
                                                   geometry::TwoPointFiniteVolumeGeometry) where D
    T = float_type(mesh)
    subcells = SubCell{D, T}[]
    
    # Get cell centroid
    cell_centroid = SVector{D, T}(geometry.cell_centroids[:, cell])
    
    # Get faces of this cell
    cell_faces = mesh.faces.cells_to_faces[cell]
    
    # For each face, create sub-cells connecting the face to the cell centroid
    for face_idx in cell_faces
        face = mesh.faces.faces_to_nodes[face_idx]
        face_centroid = SVector{D, T}(geometry.face_centroids[:, face_idx])
        
        # Determine flux orientation (into or out of cell)
        neighbors = mesh.faces.neighbors[face_idx]
        if neighbors[1] == cell
            flux_sign = 1.0
        else
            flux_sign = -1.0
        end
        flux = flux_sign * fluxes[face_idx]
        
        # Get face area and normal
        area = geometry.areas[face_idx]
        normal = SVector{D, T}(geometry.normals[:, face_idx])
        
        if D == 3
            # 3D: Create tetrahedra from face triangles
            # First, triangulate the face from its nodes
            face_triangles = triangulate_face_nodes(mesh, face)
            
            for triangle in face_triangles
                # Create a tetrahedron from triangle + cell centroid
                vertices = [
                    mesh.node_points[triangle[1]],
                    mesh.node_points[triangle[2]],
                    mesh.node_points[triangle[3]],
                    cell_centroid
                ]
                
                subcell_centroid, measure = compute_tet_centroid_and_volume(vertices)
                
                # Reconstruct velocity: flux-weighted contribution
                # Velocity is reconstructed as flux/area in the normal direction
                # distributed based on sub-cell contribution to total face
                velocity = reconstruct_subcell_velocity(flux, area, normal, measure, geometry.volumes[cell])
                
                push!(subcells, SubCell(cell, vertices, subcell_centroid, velocity, measure))
            end
        elseif D == 2
            # 2D: Create triangles from edge + cell centroid
            @assert length(face) == 2 "2D faces should have 2 nodes"
            vertices = [
                mesh.node_points[face[1]],
                mesh.node_points[face[2]],
                cell_centroid
            ]
            
            subcell_centroid, measure = compute_tri_centroid_and_area(vertices)
            velocity = reconstruct_subcell_velocity(flux, area, normal, measure, geometry.volumes[cell])
            
            push!(subcells, SubCell(cell, vertices, subcell_centroid, velocity, measure))
        else
            error("Only 2D and 3D meshes are supported")
        end
    end
    
    return subcells
end

"""
    triangulate_face_nodes(mesh, face)

Triangulate a face (polygon) into triangles for tesselation.
Uses simple fan triangulation from the first node.
"""
function triangulate_face_nodes(mesh::UnstructuredMesh, face::AbstractVector{Int})
    if length(face) == 3
        # Already a triangle
        return [face]
    else
        # Fan triangulation from first vertex
        triangles = Vector{Vector{Int}}()
        for i in 2:(length(face)-1)
            push!(triangles, [face[1], face[i], face[i+1]])
        end
        return triangles
    end
end

"""
    compute_tet_centroid_and_volume(vertices)

Compute centroid and volume of a tetrahedron.
"""
function compute_tet_centroid_and_volume(vertices::Vector{SVector{3, T}}) where T
    @assert length(vertices) == 4
    # Centroid is average of vertices
    centroid = (vertices[1] + vertices[2] + vertices[3] + vertices[4]) / 4
    
    # Volume using determinant formula
    v0 = vertices[1]
    v1 = vertices[2] - v0
    v2 = vertices[3] - v0
    v3 = vertices[4] - v0
    
    volume = abs(dot(v1, cross(v2, v3))) / 6
    
    return (centroid, volume)
end

"""
    compute_tri_centroid_and_area(vertices)

Compute centroid and area of a triangle.
"""
function compute_tri_centroid_and_area(vertices::Vector{SVector{2, T}}) where T
    @assert length(vertices) == 3
    # Centroid is average of vertices
    centroid = (vertices[1] + vertices[2] + vertices[3]) / 3
    
    # Area using cross product
    v1 = vertices[2] - vertices[1]
    v2 = vertices[3] - vertices[1]
    
    area = abs(v1[1] * v2[2] - v1[2] * v2[1]) / 2
    
    return (centroid, area)
end

"""
    reconstruct_subcell_velocity(flux, face_area, normal, subcell_measure, cell_volume)

Reconstruct velocity in a sub-cell from face flux.
The velocity is computed as the flux-weighted contribution in the normal direction.
"""
function reconstruct_subcell_velocity(flux::T, face_area::T, normal::SVector{D, T}, 
                                       subcell_measure::T, cell_volume::T) where {D, T}
    # Simple reconstruction: velocity = (flux / area) * normal * (subcell_measure / cell_volume)
    # This distributes the flux proportional to sub-cell size
    flux_velocity = (flux / face_area) * normal
    
    # Weight by sub-cell contribution
    weight = subcell_measure / cell_volume
    
    return flux_velocity * weight
end

"""
    compute_global_bbox(mesh)

Compute the global bounding box of the mesh.
"""
function compute_global_bbox(mesh::UnstructuredMesh{D}) where D
    T = float_type(mesh)
    bbox_min = fill(typemax(T), D)
    bbox_max = fill(typemin(T), D)
    
    for pt in mesh.node_points
        for d in 1:D
            bbox_min[d] = min(bbox_min[d], pt[d])
            bbox_max[d] = max(bbox_max[d], pt[d])
        end
    end
    
    return (SVector{D, T}(bbox_min), SVector{D, T}(bbox_max))
end

"""
    build_octree(subcells, bbox_min, bbox_max, max_depth)

Build an octree spatial index for sub-cells.
"""
function build_octree(subcells::Vector{SubCell{D, T}}, bbox_min::SVector{D, T}, 
                      bbox_max::SVector{D, T}, max_depth::Int) where {D, T}
    # Start with all sub-cells
    indices = collect(1:length(subcells))
    return build_octree_recursive(subcells, indices, bbox_min, bbox_max, max_depth, 0)
end

"""
    build_octree_recursive(subcells, indices, bbox_min, bbox_max, max_depth, current_depth)

Recursively build octree nodes.
"""
function build_octree_recursive(subcells::Vector{SubCell{D, T}}, indices::Vector{Int},
                                bbox_min::SVector{D, T}, bbox_max::SVector{D, T}, 
                                max_depth::Int, current_depth::Int) where {D, T}
    # Create leaf node if we've reached max depth or have few enough sub-cells
    if current_depth >= max_depth || length(indices) <= 8
        return OctreeNode(bbox_min, bbox_max, nothing, indices, 0)
    end
    
    # Otherwise, subdivide
    center = (bbox_min + bbox_max) / 2
    num_children = D == 3 ? 8 : 4
    
    # Create child bounding boxes and distribute sub-cells
    children = OctreeNode{D, T}[]
    
    for child_idx in 1:num_children
        child_min, child_max = get_child_bbox(bbox_min, bbox_max, center, child_idx, D)
        
        # Find sub-cells that overlap this child box
        child_indices = Int[]
        for idx in indices
            subcell = subcells[idx]
            if bbox_overlap(subcell.centroid, child_min, child_max)
                push!(child_indices, idx)
            end
        end
        
        if !isempty(child_indices)
            child = build_octree_recursive(subcells, child_indices, child_min, child_max, 
                                          max_depth, current_depth + 1)
            push!(children, child)
        end
    end
    
    # Compute max depth
    max_child_depth = isempty(children) ? 0 : maximum(c.max_depth for c in children) + 1
    
    return OctreeNode(bbox_min, bbox_max, children, Int[], max_child_depth)
end

"""
    get_child_bbox(bbox_min, bbox_max, center, child_idx, D)

Get the bounding box for a child octree node.
"""
function get_child_bbox(bbox_min::SVector{D, T}, bbox_max::SVector{D, T}, 
                        center::SVector{D, T}, child_idx::Int, ::Val{3}) where {D, T}
    # 3D octree: 8 children
    x_low = (child_idx - 1) & 1 == 0
    y_low = (child_idx - 1) & 2 == 0
    z_low = (child_idx - 1) & 4 == 0
    
    child_min = SVector{3, T}(
        x_low ? bbox_min[1] : center[1],
        y_low ? bbox_min[2] : center[2],
        z_low ? bbox_min[3] : center[3]
    )
    child_max = SVector{3, T}(
        x_low ? center[1] : bbox_max[1],
        y_low ? center[2] : bbox_max[2],
        z_low ? center[3] : bbox_max[3]
    )
    
    return (child_min, child_max)
end

function get_child_bbox(bbox_min::SVector{D, T}, bbox_max::SVector{D, T}, 
                        center::SVector{D, T}, child_idx::Int, ::Val{2}) where {D, T}
    # 2D quadtree: 4 children
    x_low = (child_idx - 1) & 1 == 0
    y_low = (child_idx - 1) & 2 == 0
    
    child_min = SVector{2, T}(
        x_low ? bbox_min[1] : center[1],
        y_low ? bbox_min[2] : center[2]
    )
    child_max = SVector{2, T}(
        x_low ? center[1] : bbox_max[1],
        y_low ? center[2] : bbox_max[2]
    )
    
    return (child_min, child_max)
end

function get_child_bbox(bbox_min::SVector{D, T}, bbox_max::SVector{D, T}, 
                        center::SVector{D, T}, child_idx::Int, D_int::Int) where {D, T}
    return get_child_bbox(bbox_min, bbox_max, center, child_idx, Val{D_int}())
end

"""
    bbox_overlap(point, bbox_min, bbox_max)

Check if a point is within a bounding box (with small tolerance).
"""
function bbox_overlap(point::SVector{D, T}, bbox_min::SVector{D, T}, 
                      bbox_max::SVector{D, T}; tol::T = T(1e-10)) where {D, T}
    for d in 1:D
        if point[d] < bbox_min[d] - tol || point[d] > bbox_max[d] + tol
            return false
        end
    end
    return true
end

"""
    find_subcell_at_point(tracer::StreamlineTracer, point)

Find the sub-cell containing a given point using the octree.
"""
function find_subcell_at_point(tracer::StreamlineTracer{D, T}, point::SVector{D, T}) where {D, T}
    return find_subcell_at_point_recursive(tracer.subcells, tracer.octree, point)
end

"""
    find_subcell_at_point_recursive(subcells, node, point)

Recursively search octree for sub-cell containing point.
"""
function find_subcell_at_point_recursive(subcells::Vector{SubCell{D, T}}, 
                                        node::OctreeNode{D, T}, 
                                        point::SVector{D, T}) where {D, T}
    # Check if point is in this node's bounding box
    if !bbox_overlap(point, node.bbox_min, node.bbox_max)
        return nothing
    end
    
    # If leaf node, check sub-cells
    if isnothing(node.children)
        for idx in node.subcell_indices
            subcell = subcells[idx]
            if point_in_subcell(point, subcell)
                return idx
            end
        end
        return nothing
    end
    
    # Otherwise, recurse into children
    for child in node.children
        result = find_subcell_at_point_recursive(subcells, child, point)
        if !isnothing(result)
            return result
        end
    end
    
    return nothing
end

"""
    point_in_subcell(point, subcell)

Check if a point is inside a sub-cell using barycentric coordinates.
"""
function point_in_subcell(point::SVector{3, T}, subcell::SubCell{3, T}) where T
    # For tetrahedron, use barycentric coordinates
    v = subcell.vertices
    @assert length(v) == 4
    
    # Compute barycentric coordinates
    v0 = v[1]
    v1 = v[2] - v0
    v2 = v[3] - v0
    v3 = v[4] - v0
    vp = point - v0
    
    # Solve for barycentric coordinates
    # This is a simplified check - could be more robust
    det = dot(v1, cross(v2, v3))
    if abs(det) < 1e-10
        return false
    end
    
    λ1 = dot(vp, cross(v2, v3)) / det
    λ2 = dot(v1, cross(vp, v3)) / det
    λ3 = dot(v1, cross(v2, vp)) / det
    λ0 = 1 - λ1 - λ2 - λ3
    
    tol = -1e-6
    return λ0 >= tol && λ1 >= tol && λ2 >= tol && λ3 >= tol
end

function point_in_subcell(point::SVector{2, T}, subcell::SubCell{2, T}) where T
    # For triangle, use barycentric coordinates
    v = subcell.vertices
    @assert length(v) == 3
    
    v0 = v[1]
    v1 = v[2]
    v2 = v[3]
    
    # Compute barycentric coordinates
    denom = (v1[2] - v2[2]) * (v0[1] - v2[1]) + (v2[1] - v1[1]) * (v0[2] - v2[2])
    if abs(denom) < 1e-10
        return false
    end
    
    λ0 = ((v1[2] - v2[2]) * (point[1] - v2[1]) + (v2[1] - v1[1]) * (point[2] - v2[2])) / denom
    λ1 = ((v2[2] - v0[2]) * (point[1] - v2[1]) + (v0[1] - v2[1]) * (point[2] - v2[2])) / denom
    λ2 = 1 - λ0 - λ1
    
    tol = -1e-6
    return λ0 >= tol && λ1 >= tol && λ2 >= tol
end

"""
    trace_streamlines(tracer::StreamlineTracer, start_points; 
                     max_steps = 1000, step_size = 0.1, 
                     forward = true, backward = false)

Tracing phase: Compute streamlines from given starting points.

# Arguments
- `tracer`: Pre-computed StreamlineTracer from setup phase
- `start_points`: Vector of starting points (as SVector or Matrix)
- `max_steps`: Maximum number of integration steps per streamline
- `step_size`: Integration step size
- `forward`: Trace in forward direction (along velocity)
- `backward`: Trace in backward direction (against velocity)

# Returns
A vector of streamlines, where each streamline is a vector of points.
"""
function trace_streamlines(tracer::StreamlineTracer{D, T}, 
                          start_points::AbstractVector{SVector{D, T}};
                          max_steps::Int = 1000,
                          step_size::T = T(0.1),
                          forward::Bool = true,
                          backward::Bool = false) where {D, T}
    streamlines = Vector{Vector{SVector{D, T}}}()
    
    for start_point in start_points
        streamline = SVector{D, T}[]
        
        if forward
            line = trace_single_streamline(tracer, start_point, step_size, max_steps, T(1))
            append!(streamline, line)
        end
        
        if backward
            line = trace_single_streamline(tracer, start_point, step_size, max_steps, T(-1))
            # Reverse and append (excluding start point to avoid duplication)
            append!(streamline, reverse(line[2:end]))
        end
        
        push!(streamlines, streamline)
    end
    
    return streamlines
end

# Overload for Matrix input
function trace_streamlines(tracer::StreamlineTracer{D, T}, 
                          start_points::AbstractMatrix;
                          kwarg...) where {D, T}
    # Convert matrix to vector of SVectors
    @assert size(start_points, 1) == D "Start points matrix must have $D rows"
    points = [SVector{D, T}(start_points[:, i]) for i in 1:size(start_points, 2)]
    return trace_streamlines(tracer, points; kwarg...)
end

"""
    trace_single_streamline(tracer, start_point, step_size, max_steps, direction)

Trace a single streamline using Euler integration.
"""
function trace_single_streamline(tracer::StreamlineTracer{D, T}, 
                                start_point::SVector{D, T},
                                step_size::T,
                                max_steps::Int,
                                direction::T) where {D, T}
    points = SVector{D, T}[start_point]
    current_point = start_point
    
    for step in 1:max_steps
        # Find sub-cell containing current point
        subcell_idx = find_subcell_at_point(tracer, current_point)
        
        if isnothing(subcell_idx)
            # Left the domain
            break
        end
        
        subcell = tracer.subcells[subcell_idx]
        velocity = subcell.velocity
        
        # Check if velocity is nearly zero (stagnation point)
        speed = norm(velocity, 2)
        if speed < 1e-10
            break
        end
        
        # Euler step
        next_point = current_point + direction * step_size * velocity / speed
        
        push!(points, next_point)
        current_point = next_point
        
        # Check for very small movement (potential cycle or stagnation)
        if step > 1 && norm(current_point - points[end-1], 2) < 1e-10
            break
        end
    end
    
    return points
end

# Helper to convert various start point formats
function convert_start_points(start_points::AbstractMatrix{T}, ::Val{D}) where {D, T}
    @assert size(start_points, 1) == D
    return [SVector{D, T}(start_points[:, i]) for i in 1:size(start_points, 2)]
end

function convert_start_points(start_points::Vector{SVector{D, T}}, ::Val{D}) where {D, T}
    return start_points
end
