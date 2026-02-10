#!/usr/bin/env julia
"""
Demonstration of PEBI/Voronoi Mesh functionality in Jutul.jl

This script shows how to create and use PEBI meshes.
"""

using Jutul

println("\n" * "=" ^ 70)
println("PEBI/Voronoi Mesh Demonstration")
println("=" ^ 70)

# Example 1: Simple square with 4 points
println("\n[Example 1] Creating a simple 2D PEBI mesh with 4 points")
println("-" ^ 70)

points = [0.0 1.0 0.0 1.0; 
          0.0 0.0 1.0 1.0]

println("Input points (2×4 matrix):")
println("  Point 1: (0.0, 0.0)")
println("  Point 2: (1.0, 0.0)")
println("  Point 3: (0.0, 1.0)")
println("  Point 4: (1.0, 1.0)")

mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)

println("\nMesh created successfully!")
println("Mesh type: $(typeof(mesh))")
println("\nMesh statistics:")
println("  - Number of cells: $(number_of_cells(mesh))")
println("  - Number of nodes: $(length(mesh.node_points))")
println("  - Internal faces: $(length(mesh.faces.neighbors))")
println("  - Boundary faces: $(length(mesh.boundary_faces.neighbors))")

# Example 2: Different input format
println("\n[Example 2] Creating mesh with alternative input format")
println("-" ^ 70)

points_tuples = [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5)]
mesh2 = Jutul.VoronoiMeshes.PEBIMesh2D(points_tuples)

println("Input format: Vector of tuples")
println("Points:")
for (i, p) in enumerate(points_tuples)
    println("  Point $i: $p")
end

println("\nMesh created successfully!")
println("  - Cells: $(number_of_cells(mesh2))")
println("  - Nodes: $(length(mesh2.node_points))")

# Example 3: Accessing mesh properties
println("\n[Example 3] Accessing mesh properties")
println("-" ^ 70)

mesh = mesh1 = Jutul.VoronoiMeshes.PEBIMesh2D([0.0 1.0 0.0 1.0; 
                                               0.0 0.0 1.0 1.0])

println("Node coordinates:")
for (i, node) in enumerate(mesh.node_points)
    println("  Node $i: $(round.(node, digits=3))")
end

println("\nCell-to-face mappings:")
for cell_idx in 1:number_of_cells(mesh)
    faces = mesh.faces.cells_to_faces[cell_idx]
    println("  Cell $cell_idx: faces $faces")
end

println("\nFace neighbors (internal faces):")
for (face_idx, (c1, c2)) in enumerate(mesh.faces.neighbors)
    println("  Face $face_idx: cells ($c1, $c2)")
end

println("\nBoundary faces:")
for (face_idx, cell) in enumerate(mesh.boundary_faces.neighbors)
    println("  Boundary face $face_idx: cell $cell")
end

# Example 4: 3D mesh (shows error handling)
println("\n[Example 4] Testing error handling for 3D mesh")
println("-" ^ 70)

points_3d = [0.0 1.0 0.0 1.0;
             0.0 0.0 1.0 1.0;
             0.0 0.0 0.0 1.0]

try
    mesh_3d = Jutul.VoronoiMeshes.PEBIMesh3D(points_3d)
catch e
    println("Expected error caught:")
    println("  Error type: $(typeof(e).name.name)")
    println("  Message: $(string(e)[1:70])...")
end

println("\n" * "=" ^ 70)
println("Demonstration complete!")
println("=" ^ 70 * "\n")
