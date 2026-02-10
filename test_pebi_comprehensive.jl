#!/usr/bin/env julia
"""
Comprehensive test for PEBI/Voronoi mesh implementation in Jutul.jl
Tests:
1. Simple 2D mesh with 4 points
2. Various point input formats
3. Mesh connectivity validation
4. Geometry checks
"""

using Jutul

println("=" ^ 60)
println("PEBI/Voronoi Mesh Implementation Tests")
println("=" ^ 60)

# Test 1: Simple 2D mesh with 4 points (square)
println("\n[Test 1] Creating 2D PEBI mesh with 4 points (square grid)...")
points_matrix = [0.0 1.0 0.0 1.0; 
                 0.0 0.0 1.0 1.0]
mesh1 = Jutul.VoronoiMeshes.PEBIMesh2D(points_matrix)
@assert mesh1 isa Jutul.UnstructuredMesh "Mesh should be UnstructuredMesh"
@assert number_of_cells(mesh1) == 4 "Should have 4 cells"
println("  ✓ Successfully created 4-point mesh")
println("    - Cells: $(number_of_cells(mesh1))")
println("    - Nodes: $(length(mesh1.node_points))")
println("    - Internal faces: $(length(mesh1.faces.neighbors))")
println("    - Boundary faces: $(length(mesh1.boundary_faces.neighbors))")

# Test 2: Vector of tuples format
println("\n[Test 2] Creating mesh with vector of tuples format...")
points_tuples = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
mesh2 = Jutul.VoronoiMeshes.PEBIMesh2D(points_tuples)
@assert mesh2 isa Jutul.UnstructuredMesh "Mesh should be UnstructuredMesh"
@assert number_of_cells(mesh2) == 4 "Should have 4 cells"
println("  ✓ Successfully created mesh from tuple format")

# Test 3: Vector of vectors format
println("\n[Test 3] Creating mesh with vector of vectors format...")
points_vectors = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
mesh3 = Jutul.VoronoiMeshes.PEBIMesh2D(points_vectors)
@assert mesh3 isa Jutul.UnstructuredMesh "Mesh should be UnstructuredMesh"
@assert number_of_cells(mesh3) == 4 "Should have 4 cells"
println("  ✓ Successfully created mesh from vector format")

# Test 4: Transposed matrix format (n x 2)
println("\n[Test 4] Creating mesh with transposed matrix format (n x 2)...")
points_transposed = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
mesh4 = Jutul.VoronoiMeshes.PEBIMesh2D(points_transposed)
@assert mesh4 isa Jutul.UnstructuredMesh "Mesh should be UnstructuredMesh"
@assert number_of_cells(mesh4) == 4 "Should have 4 cells"
println("  ✓ Successfully created mesh from transposed matrix format")

# Test 5: Mesh connectivity validation
println("\n[Test 5] Validating mesh connectivity...")
mesh = mesh1
nc = number_of_cells(mesh)

# Check that cell neighbors are valid
for cell_idx in 1:nc
    cell_faces = mesh.faces.cells_to_faces[cell_idx]
    @assert all(f -> 1 <= f <= length(mesh.faces.neighbors), cell_faces) "Invalid face indices"
end

# Check that face neighbors are valid
for (c1, c2) in mesh.faces.neighbors
    @assert 1 <= c1 <= nc "Invalid cell index in neighbor"
    @assert 1 <= c2 <= nc "Invalid cell index in neighbor"
end

# Check that boundary neighbors are valid
for c_bnd in mesh.boundary_faces.neighbors
    @assert 1 <= c_bnd <= nc "Invalid cell index in boundary neighbor"
end

println("  ✓ Mesh connectivity is valid")
println("    - All face neighbors are valid")
println("    - All boundary neighbors are valid")

# Test 6: Geometry validation (node points)
println("\n[Test 6] Validating mesh geometry...")
mesh = mesh1
nodes = mesh.node_points

# All nodes should be 2D
@assert all(n -> length(n) == 2, nodes) "All nodes should be 2D"

# Check that nodes are finite
@assert all(n -> all(isfinite.(n)), nodes) "All node coordinates should be finite"

println("  ✓ Mesh geometry is valid")
println("    - All $(length(nodes)) nodes are 2D")
println("    - All coordinates are finite")

# Print node bounds
xs = [n[1] for n in nodes]
ys = [n[2] for n in nodes]
println("    - X range: [$(minimum(xs)), $(maximum(xs))]")
println("    - Y range: [$(minimum(ys)), $(maximum(ys))]")

# Test 7: 3D mesh error handling
println("\n[Test 7] Testing 3D mesh error handling...")
points_3d = [0.0 1.0 0.0 1.0;
             0.0 0.0 1.0 1.0;
             0.0 0.0 0.0 0.0]
try
    mesh_3d = Jutul.VoronoiMeshes.PEBIMesh3D(points_3d)
    println("  ✗ FAILED: Should have thrown an error for 3D mesh")
    exit(1)
catch e
    if contains(string(e), "not yet implemented")
        println("  ✓ Correctly throws error for unimplemented 3D mesh")
        error_str = string(e)
        if length(error_str) > 80
            error_str = error_str[1:80] * "..."
        end
        println("    - Error message: $error_str")
    else
        println("  ✗ FAILED: Wrong error type: ", e)
        exit(1)
    end
end

println("\n" * ("=" ^ 60))
println("✓ All tests passed successfully!")
println("=" ^ 60)
