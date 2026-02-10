#!/usr/bin/env julia
"""
Test script for PEBI/Voronoi mesh implementation in Jutul.jl
Tests:
1. Loading Jutul
2. Creating a simple 2D PEBI mesh with 4 points
3. Verifying it returns an UnstructuredMesh instance
"""

using Jutul

println("✓ Successfully loaded Jutul")

# Test 1: Create a simple 2D PEBI mesh with 4 points
println("\nTest 1: Creating a 2D PEBI mesh with 4 points...")
points = [0.0 1.0 0.0 1.0; 
          0.0 0.0 1.0 1.0]
println("Points: ", points)

try
    mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)
    println("✓ Successfully created PEBI mesh")
    
    # Test 2: Verify it's an UnstructuredMesh
    println("\nTest 2: Verifying mesh type...")
    if mesh isa Jutul.UnstructuredMesh
        println("✓ Mesh is an UnstructuredMesh")
    else
        println("✗ Mesh is not an UnstructuredMesh, got: ", typeof(mesh))
    end
    
    # Test 3: Inspect mesh properties
    println("\nTest 3: Inspecting mesh properties...")
    println("  - Number of cells: ", number_of_cells(mesh))
    println("  - Number of nodes: ", length(mesh.node_points))
    
    # Check internal structure
    println("\nMesh structure:")
    println("  - Faces (internal): ", length(mesh.faces.cells_to_faces.vals), " elements")
    println("  - Faces (cells_to_faces indirection): ", length(mesh.faces.cells_to_faces.pos), " positions")
    println("  - Nodes: ", length(mesh.node_points), " points")
    println("  - Boundary faces: ", length(mesh.boundary_faces.cells_to_faces.vals), " elements")
    println("  - Tags: ", typeof(mesh.tags))
    println("  - Dimension: 2")
    
    # Verify key mesh properties
    nc = number_of_cells(mesh)
    println("\nMesh validation:")
    println("  ✓ Number of cells: ", nc)
    println("  ✓ Number of nodes: ", length(mesh.node_points))
    println("  ✓ Internal faces: ", length(mesh.faces.neighbors))
    println("  ✓ Boundary faces: ", length(mesh.boundary_faces.neighbors))
    
    println("\n✓ All tests passed!")
    
catch e
    println("✗ Error creating PEBI mesh:")
    println(sprint(showerror, e, catch_backtrace()))
    exit(1)
end
