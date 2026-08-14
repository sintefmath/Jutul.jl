# Streamline Tracing

The streamline tracing functionality allows you to compute flow pathways through an `UnstructuredMesh` based on face fluxes. The implementation uses a two-phase approach:

1. **Setup Phase**: Preprocesses the mesh and flux field
2. **Tracing Phase**: Fast computation of streamlines from arbitrary starting points

## Overview

The streamline tracer works by:
1. Subdividing each cell into tetrahedral (3D) or triangular (2D) sub-cells using centroid tesselation
2. Reconstructing velocity fields in each sub-cell from face fluxes
3. Building an octree spatial index for fast point location
4. Tracing streamlines with Euler, RK2, or RK4 integration

## Basic Usage

```julia
using Jutul

# Create or load a mesh
mesh = UnstructuredMesh(...)

# Compute or obtain face fluxes (one value per interior face)
fluxes = ... # Vector{Float64} of length number_of_faces(mesh)

# Setup phase (done once)
tracer = setup_streamline_tracer(mesh, fluxes)

# Tracing phase (can be called many times with different starting points)
start_points = [SVector{3, Float64}(x, y, z) for ...]
streamlines = trace_streamlines(tracer, start_points)
```

## Setup Phase

The `setup_streamline_tracer` function performs the preprocessing:

```julia
tracer = setup_streamline_tracer(
    mesh,           # UnstructuredMesh
    fluxes;         # Vector of face fluxes
    geometry = tpfv_geometry(mesh),  # Optional: pre-computed geometry
    max_depth = 8,  # Maximum octree depth
    boundary_fluxes = nothing # Optional boundary face fluxes (defaults to zero)
)
```

This function:
- Subdivides each cell into sub-cells based on centroid tesselation
- Reconstructs velocities in each sub-cell from face fluxes
- Builds an octree for efficient spatial queries

The returned `StreamlineTracer` object contains all preprocessed data and can be reused for tracing multiple sets of streamlines.

## Tracing Phase

The `trace_streamlines` function computes streamlines:

```julia
streamlines = trace_streamlines(
    tracer,              # StreamlineTracer from setup phase
    start_points;        # Starting points (Vector of SVector or Matrix)
    max_steps = 1000,    # Maximum integration steps
    step_size = 0.1,     # Integration step size
    forward = true,      # Trace forward (along velocity)
    backward = false     # Trace backward (against velocity)
)
```

### Starting Points

Starting points can be provided as:
- A `Vector{SVector{D, T}}` where D is the mesh dimension
- A `Matrix{T}` of size `D × n` where n is the number of starting points

### Return Value

Returns a `Vector{Vector{SVector{D, T}}}` where each element is a streamline (sequence of points).

## Examples

### Example 1: Single Streamline from Cell Centroid

```julia
# Get geometry
geo = tpfv_geometry(mesh)

# Start from a cell centroid
start_point = SVector{3, Float64}(geo.cell_centroids[:, 10])

# Trace streamline
streamlines = trace_streamlines(tracer, [start_point])

# Access the streamline points
points = streamlines[1]
```

### Example 2: Multiple Streamlines

```julia
# Create multiple starting points
n_streamlines = 10
start_points = [SVector{3, Float64}(randn(3)...) for i in 1:n_streamlines]

# Trace all streamlines (computed independently)
streamlines = trace_streamlines(tracer, start_points, max_steps = 500)
```

### Example 3: Forward and Backward Tracing

```julia
# Trace both forward and backward from a point
start_point = SVector{3, Float64}(0.5, 0.5, 0.5)

streamlines = trace_streamlines(
    tracer, 
    [start_point],
    forward = true,
    backward = true
)

# The returned streamline includes both directions
full_streamline = streamlines[1]
```

### Example 4: Using Matrix Input

```julia
# Starting points as a 3 × n matrix
start_matrix = randn(3, 5)

streamlines = trace_streamlines(tracer, start_matrix)
```

## Implementation Details

### Cell Tesselation

Each cell is subdivided into sub-cells by connecting each face to the cell centroid:
- In 3D: Each polygonal face is triangulated, and tetrahedra are formed from each triangle + cell centroid
- In 2D: Triangles are formed from each edge + cell centroid

Both interior and boundary faces are included so the sub-cells cover the full cell volume/area.

### Velocity Reconstruction

Velocity is reconstructed per cell from the available face fluxes by solving a small least-squares problem:
```
minimize  Σ_faces (area(face) * (n(face) ⋅ velocity) - flux(face))^2
```

The resulting cell velocity is then used in all sub-cells belonging to that cell. This makes the traced paths depend on the full local flux pattern instead of only a single face normal.

### Octree Spatial Index

An octree (or quadtree in 2D) is built to enable fast point location queries. The octree:
- Recursively subdivides space into 8 (3D) or 4 (2D) children
- Stores sub-cell indices at leaf nodes
- Has a maximum depth to limit subdivision

### Streamline Integration

Streamlines are computed with `EulerIntegrator()`, `RK2Integrator()`, or `RK4Integrator()`. All methods advance along the normalized local velocity direction:
```
next_point = current_point + step_size × (velocity / ||velocity||)
```

Tracing stops when:
- Maximum number of steps is reached
- The point leaves the domain (no containing sub-cell found)
- A stagnation point is encountered (velocity ≈ 0)
- Movement becomes negligible (potential cycle)

## Performance Considerations

- **Setup Phase**: O(n_cells × n_faces_per_cell) for tesselation, O(n_subcells × log(n_subcells)) for octree
- **Tracing Phase**: O(n_steps × log(n_subcells)) per streamline

The setup phase is more expensive but only needs to be done once per mesh/flux combination. The tracing phase is fast and can handle many starting points efficiently.

## Limitations and Future Improvements

Current limitations:
- Velocity reconstruction is simplified (could use more sophisticated interpolation)
- No adaptive step sizing
- No cycle detection beyond simple checks

Possible improvements:
- Add adaptive step sizing based on local velocity gradient
- Implement proper cycle detection
- Support for time-varying velocity fields
- Parallel tracing of multiple streamlines
