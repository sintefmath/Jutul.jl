"""
Octree spatial indexing for fast point location
"""

"""
    compute_global_bbox(mesh)

Compute the global bounding box of the mesh.
"""
function compute_global_bbox(mesh::UnstructuredMesh{D}) where D
    T = Jutul.float_type(mesh)
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
            if bbox_intersects(subcell.bbox_min, subcell.bbox_max, child_min, child_max)
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

function bbox_intersects(a_min::SVector{D, T}, a_max::SVector{D, T},
                         b_min::SVector{D, T}, b_max::SVector{D, T};
                         tol::T = T(1e-10)) where {D, T}
    for d in 1:D
        if a_max[d] < b_min[d] - tol || b_max[d] < a_min[d] - tol
            return false
        end
    end
    return true
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
