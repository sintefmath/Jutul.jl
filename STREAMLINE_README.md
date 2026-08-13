# Streamline Tracing Implementation for UnstructuredMesh

This implementation adds streamline tracing functionality to the Jutul.jl package, specifically for the `UnstructuredMesh` type.

## Architecture

The implementation follows a two-phase design as requested:

### Phase 1: Setup (Preprocessing)
- **Cell Tesselation**: Each cell is subdivided into tetrahedral (3D) or triangular (2D) sub-cells using centroid-based tesselation
- **Velocity Reconstruction**: Velocities are reconstructed in each sub-cell from face fluxes
- **Octree Construction**: An octree (or quadtree in 2D) spatial index is built for fast point location queries

### Phase 2: Tracing (Fast Queries)
- **Point Location**: Fast lookup of sub-cells using the octree
- **Integration**: Euler integration along velocity field
- **Multi-directional**: Supports forward, backward, or bidirectional tracing

## Files Added

1. **src/meshes/streamlines.jl** - Main implementation
   - `StreamlineTracer` struct
   - `setup_streamline_tracer()` - Setup phase
   - `trace_streamlines()` - Tracing phase
   - `OctreeNode` - Spatial index structure
   - `SubCell` - Tesselated sub-cell structure

2. **test/streamlines.jl** - Comprehensive test suite
   - Tests for 2D and 3D meshes
   - Point location tests
   - Forward/backward tracing tests
   - Multiple starting points tests
   - Geometric computation tests

3. **docs/streamline_tracing.md** - Documentation
   - Usage examples
   - API reference
   - Implementation details
   - Performance considerations

4. **examples/streamline_example.jl** - Example usage
   - Complete working example
   - Demonstrates typical workflow

## Key Features

✅ **Centroid Tesselation**: Cells are subdivided by connecting faces to cell centroids
✅ **Octree Spatial Index**: O(log n) point location queries
✅ **Velocity Reconstruction**: From face fluxes using flux conservation
✅ **Bidirectional Tracing**: Forward and/or backward from starting points
✅ **Multiple Starting Points**: Efficient batch processing
✅ **Flexible Input**: Supports both SVector and Matrix inputs
✅ **2D and 3D Support**: Works with both 2D and 3D meshes

## API Summary

```julia
# Setup phase (done once per mesh/flux combination)
tracer = setup_streamline_tracer(mesh, fluxes; geometry, max_depth)

# Tracing phase (can be called many times)
streamlines = trace_streamlines(tracer, start_points; 
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
- Simple Euler integration: `next = current + step_size × (velocity / ||velocity||)`
- Stops on: max steps, domain exit, stagnation, or negligible movement

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
- Higher-order integration (Runge-Kutta)
- Adaptive step sizing
- Time-varying velocity fields
- Parallel tracing
- More sophisticated velocity interpolation
- Cycle detection
- Streamtube computation

## Integration with Jutul

The implementation integrates seamlessly with existing Jutul components:
- Uses `UnstructuredMesh` type
- Compatible with `tpfv_geometry`
- Works with standard face flux computations
- Follows Jutul coding conventions
- Uses StaticArrays for performance
