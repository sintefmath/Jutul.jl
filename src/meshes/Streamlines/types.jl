"""
Core types for streamline tracing
"""

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
mutable struct SubCell{D, T}
    "Parent cell index in the mesh"
    parent_cell::Int
    "Parent face index in the mesh"
    parent_face::Int
    "True if the parent face is a boundary face"
    parent_face_is_boundary::Bool
    "Vertex coordinates"
    vertices::Vector{SVector{D, T}}
    "Centroid of the sub-cell"
    centroid::SVector{D, T}
    "Reconstructed velocity at the centroid"
    velocity::SVector{D, T}
    "Volume/area of the sub-cell"
    measure::T
    "Bounding box minimum coordinates"
    bbox_min::SVector{D, T}
    "Bounding box maximum coordinates"
    bbox_max::SVector{D, T}
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
    StreamlineIntegrator

Abstract type for streamline integration methods.
"""
abstract type StreamlineIntegrator end

"""
    EulerIntegrator <: StreamlineIntegrator

Forward Euler integration method (1st order).
Simple and fast but less accurate than higher-order methods.
"""
struct EulerIntegrator <: StreamlineIntegrator end

"""
    RK2Integrator <: StreamlineIntegrator

2nd-order Runge-Kutta integration (Heun's method / improved Euler).
Better accuracy than Euler with moderate computational cost.
"""
struct RK2Integrator <: StreamlineIntegrator end

"""
    RK4Integrator <: StreamlineIntegrator

4th-order classical Runge-Kutta integration.
High accuracy but requires 4 velocity evaluations per step.
"""
struct RK4Integrator <: StreamlineIntegrator end
