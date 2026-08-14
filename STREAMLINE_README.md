# Streamline Tracing Module for UnstructuredMesh

This implementation adds streamline tracing functionality to the Jutul.jl package, specifically for the `UnstructuredMesh` type. The functionality is now organized in a dedicated `Streamlines` module.

## Architecture

The implementation follows a two-phase design as requested:

### Phase 1: Setup (Preprocessing)
- **Cell Tesselation**: Each cell is subdivided into tetrahedral (3D) or triangular (2D) sub-cells using centroid-based tesselation
- **Velocity Reconstruction**: Velocities are reconstructed in each sub-cell from face fluxes
- **Octree Construction**: An octree (or quadtree in 2D) spatial index is built for fast point location queries

### Phase 2: Tracing (Fast Queries)
- **Point Location**: Fast lookup of sub-cells using the octree
- **Integration**: Multiple integration methods available (Euler, RK2, RK4)
- **Multi-directional**: Supports forward, backward, or bidirectional tracing

## Module Structure

The streamline functionality is now organized in `src/meshes/Streamlines/` with the following files:

1. **Streamlines.jl** - Main module definition
2. **types.jl** - Core data structures (StreamlineTracer, SubCell, OctreeNode, integrators)
3. **integrators.jl** - Integration methods (Euler, RK2, RK4)
4. **octree.jl** - Spatial indexing and point location
5. **tesselation.jl** - Cell subdivision and velocity reconstruction
6. **tracing.jl** - Main streamline tracing functionality

2. **test/streamlines.jl** - Comprehensive test suite
   - Tests for 2D and 3D meshes
   - Point location tests
   - Forward/backward tracing tests
   - Multiple starting points tests
   - Geometric computation tests
   - Integration method tests (Euler, RK2, RK4)

3. **docs/streamline_tracing.md** - Documentation
   - Usage examples
   - API reference
   - Implementation details
   - Performance considerations

4. **examples/streamline_example.jl** - Example usage
   - Complete working example
   - Demonstrates typical workflow
   - Compares different integration methods

## Integration Methods

The module now supports three integration methods with varying accuracy:

### Euler Integrator (1st order)
```julia
streamlines = trace_streamlines(tracer, start_points, integrator = EulerIntegrator())
```
- Fastest, simplest method
- First-order accuracy: O(h)
- Good for quick visualizations

### RK2 Integrator (2nd order - Heun's method)
```julia
streamlines = trace_streamlines(tracer, start_points, integrator = RK2Integrator())
```
- Balance between speed and accuracy
- Second-order accuracy: O(h²)
- Uses 2 velocity evaluations per step

### RK4 Integrator (4th order)
```julia
streamlines = trace_streamlines(tracer, start_points, integrator = RK4Integrator())
```
- Highest accuracy
- Fourth-order accuracy: O(h⁴)
- Uses 4 velocity evaluations per step
- Best for smooth, accurate streamlines

## Key Features

✅ **Module Organization**: Clean separation into dedicated Streamlines module
✅ **Centroid Tesselation**: Cells are subdivided by connecting faces to cell centroids
✅ **Octree Spatial Index**: O(log n) point location queries
✅ **Velocity Reconstruction**: From face fluxes using flux conservation
✅ **Multiple Integrators**: Euler (1st), RK2 (2nd), and RK4 (4th) order methods
✅ **Bidirectional Tracing**: Forward and/or backward from starting points
✅ **Multiple Starting Points**: Efficient batch processing
✅ **Flexible Input**: Supports both SVector and Matrix inputs
✅ **2D and 3D Support**: Works with both 2D and 3D meshes

## API Summary

```julia
# Import the module
using Jutul
using Jutul.Streamlines

# Setup phase (done once per mesh/flux combination)
tracer = setup_streamline_tracer(mesh, fluxes; geometry, max_depth)

# Tracing phase (can be called many times with different integrators)
# Default: Euler integrator
streamlines = trace_streamlines(tracer, start_points; 
                                max_steps, step_size, 
                                forward, backward)

# Or specify integrator explicitly
streamlines = trace_streamlines(tracer, start_points;
                                integrator = RK4Integrator(),
                                max_steps, step_size, 
                                forward, backward)
```

## Implementation Details

### Cell Tesselation
- **3D**: Polygonal faces are triangulated, then tetrahedra formed with cell centroid
- **2D**: Edges form triangles with cell centroid
- Each sub-cell stores: parent cell, vertices, centroid, velocity, measure

### Velocity Reconstruction
```julia
velocity = (flux / face_area) × normal × (subcell_measure / cell_volume)
```

### Octree Structure
- Recursively subdivides space into 8 (3D) or 4 (2D) regions
- Leaf nodes store sub-cell indices
- Configurable max depth for space/time tradeoff

### Streamline Integration

Three integration methods are available:

1. **Euler**: `next = current + step_size × (velocity / ||velocity||)`
   - Stops on: max steps, domain exit, stagnation, or negligible movement

2. **RK2 (Heun)**: Two-stage Runge-Kutta
   - k1 = v(x_n)
   - k2 = v(x_n + h * k1/||k1||)
   - x_{n+1} = x_n + h * (k1 + k2) / (2 * ||(k1 + k2)||)

3. **RK4 (Classical)**: Four-stage Runge-Kutta
   - k1 = v(x_n)
   - k2 = v(x_n + h/2 * k1/||k1||)
   - k3 = v(x_n + h/2 * k2/||k2||)
   - k4 = v(x_n + h * k3/||k3||)
   - x_{n+1} = x_n + h * (k1 + 2*k2 + 2*k3 + k4) / (6 * ||...||)

## Performance

- **Setup**: O(n_cells × n_faces_per_cell) + O(n_subcells × log n_subcells)
- **Tracing**: O(n_steps × log n_subcells) per streamline

The setup phase is more expensive but only needs to be done once. The tracing phase is fast and can handle many starting points efficiently.

## Testing

Run the test suite with:
```julia
using Pkg
Pkg.test("Jutul", test_args=["streamlines"])
```

Or include in the main test suite by adding to `test/runtests.jl`.

## Future Enhancements

Potential improvements for future versions:
- ✅ ~~Higher-order integration (Runge-Kutta)~~ **DONE**
- Adaptive step sizing
- Time-varying velocity fields
- Parallel tracing
- More sophisticated velocity interpolation
- Cycle detection
- Streamtube computation

## Integration with Jutul

The implementation integrates seamlessly with existing Jutul components:
- Organized as a submodule: `Jutul.Streamlines`
- Uses `UnstructuredMesh` type
- Compatible with `tpfv_geometry`
- Works with standard face flux computations
- Follows Jutul coding conventions
- Uses StaticArrays for performance
- Extensible integrator design for adding new methods
