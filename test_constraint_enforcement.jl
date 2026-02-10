using Test
using Jutul

"""
Test constraint enforcement in PEBI mesh implementation
This test creates a simple scenario with 4 points: 
- 2 on the left side of x=0.5
- 2 on the right side of x=0.5
And enforces a vertical constraint at x=0.5
"""

@testset "PEBI Constraint Enforcement Tests" begin
    @testset "Vertical constraint at x=0.5 with 4 points" begin
        # Create points: 2 on left, 2 on right of x=0.5
        points = [0.2 0.8 0.2 0.8;
                  0.2 0.2 0.8 0.8]
        
        # Define vertical constraint at x=0.5
        constraint = ([0.5, 0.0], [0.5, 1.0])
        
        println("\n=== Testing Constraint Enforcement ===")
        println("Points:")
        println("  Point 1: (0.2, 0.2) - LEFT of constraint")
        println("  Point 2: (0.8, 0.2) - RIGHT of constraint")
        println("  Point 3: (0.2, 0.8) - LEFT of constraint")
        println("  Point 4: (0.8, 0.8) - RIGHT of constraint")
        println("Constraint: Vertical line at x=0.5")
        
        # Create mesh with constraint
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points, constraints=[constraint])
        
        @test mesh isa UnstructuredMesh
        println("\nMesh created successfully")
        println("Number of cells: $(number_of_cells(mesh))")
        println("Number of nodes: $(length(mesh.node_points))")
        
        # Get geometry for analysis
        geo = tpfv_geometry(mesh)
        @test geo isa TwoPointFiniteVolumeGeometry
        println("Geometry computed successfully")
        
        # Test constraint enforcement: cells should NOT straddle the constraint line
        violations = Int[]
        tolerance = 1e-8
        
        for cell_idx in 1:number_of_cells(mesh)
            # Collect all vertices of this cell
            cell_vertices = Set{Int}()
            
            # Internal faces
            for face_idx in mesh.faces.cells_to_faces[cell_idx]
                for node_idx in mesh.faces.faces_to_nodes[face_idx]
                    push!(cell_vertices, node_idx)
                end
            end
            
            # Boundary faces
            for face_idx in mesh.boundary_faces.cells_to_faces[cell_idx]
                for node_idx in mesh.boundary_faces.faces_to_nodes[face_idx]
                    push!(cell_vertices, node_idx)
                end
            end
            
            # Get x-coordinates of all vertices
            x_coords = [mesh.node_points[v][1] for v in cell_vertices]
            
            # Check if cell respects the constraint
            # All vertices should be on one side of x=0.5 (or exactly on it)
            all_left = all(x <= 0.5 + tolerance for x in x_coords)
            all_right = all(x >= 0.5 - tolerance for x in x_coords)
            
            # Cell violates constraint if it straddles the line
            if !(all_left || all_right)
                push!(violations, cell_idx)
                println("\n⚠️  VIOLATION: Cell $cell_idx straddles constraint")
                println("   X-coordinates of vertices: $(x_coords)")
                println("   Min X: $(minimum(x_coords))")
                println("   Max X: $(maximum(x_coords))")
            end
        end
        
        # Report results
        println("\n=== Constraint Enforcement Results ===")
        println("Total cells: $(number_of_cells(mesh))")
        println("Violations: $(length(violations))")
        
        if isempty(violations)
            println("✓ ALL CELLS RESPECT THE CONSTRAINT")
            println("✓ No cell vertices cross x=0.5 constraint line")
        else
            println("✗ CONSTRAINT ENFORCEMENT FAILED")
            println("  $(length(violations)) cell(s) cross the constraint line")
            println("  Violating cells: $(violations)")
        end
        
        @test isempty(violations) "Constraint should be enforced: $(length(violations)) cells violate it"
    end
    
    @testset "Constraint enforcement with cells on boundary" begin
        # Test case where some cells are very close to the constraint
        points = [0.2 0.78 0.22 0.79;
                  0.3 0.3  0.7  0.7]
        
        constraint = ([0.5, 0.0], [0.5, 1.0])
        
        println("\n=== Testing Constraint with Near-Boundary Points ===")
        
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points, constraints=[constraint])
        
        @test mesh isa UnstructuredMesh
        
        violations = Int[]
        tolerance = 1e-8
        
        for cell_idx in 1:number_of_cells(mesh)
            cell_vertices = Set{Int}()
            
            for face_idx in mesh.faces.cells_to_faces[cell_idx]
                for node_idx in mesh.faces.faces_to_nodes[face_idx]
                    push!(cell_vertices, node_idx)
                end
            end
            
            for face_idx in mesh.boundary_faces.cells_to_faces[cell_idx]
                for node_idx in mesh.boundary_faces.faces_to_nodes[face_idx]
                    push!(cell_vertices, node_idx)
                end
            end
            
            x_coords = [mesh.node_points[v][1] for v in cell_vertices]
            
            all_left = all(x <= 0.5 + tolerance for x in x_coords)
            all_right = all(x >= 0.5 - tolerance for x in x_coords)
            
            if !(all_left || all_right)
                push!(violations, cell_idx)
            end
        end
        
        println("Cells with constraint violations: $(length(violations))")
        if isempty(violations)
            println("✓ All cells respect the constraint")
        else
            println("✗ $(length(violations)) cell(s) violate constraint")
        end
        
        @test isempty(violations)
    end
    
    @testset "Multiple constraints" begin
        # Test with two orthogonal constraints
        points = [0.25 0.75 0.25 0.75;
                  0.25 0.25 0.75 0.75]
        
        constraint1 = ([0.5, 0.0], [0.5, 1.0])  # Vertical at x=0.5
        constraint2 = ([0.0, 0.5], [1.0, 0.5])  # Horizontal at y=0.5
        
        println("\n=== Testing Multiple Orthogonal Constraints ===")
        println("Constraints: x=0.5 and y=0.5")
        
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points, constraints=[constraint1, constraint2])
        
        @test mesh isa UnstructuredMesh
        println("Mesh created with $(number_of_cells(mesh)) cells")
        
        violations_x = Int[]
        violations_y = Int[]
        tolerance = 1e-8
        
        for cell_idx in 1:number_of_cells(mesh)
            cell_vertices = Set{Int}()
            
            for face_idx in mesh.faces.cells_to_faces[cell_idx]
                for node_idx in mesh.faces.faces_to_nodes[face_idx]
                    push!(cell_vertices, node_idx)
                end
            end
            
            for face_idx in mesh.boundary_faces.cells_to_faces[cell_idx]
                for node_idx in mesh.boundary_faces.faces_to_nodes[face_idx]
                    push!(cell_vertices, node_idx)
                end
            end
            
            coords = [mesh.node_points[v] for v in cell_vertices]
            x_coords = [c[1] for c in coords]
            y_coords = [c[2] for c in coords]
            
            # Check x constraint
            x_all_left = all(x <= 0.5 + tolerance for x in x_coords)
            x_all_right = all(x >= 0.5 - tolerance for x in x_coords)
            
            if !(x_all_left || x_all_right)
                push!(violations_x, cell_idx)
            end
            
            # Check y constraint
            y_all_bottom = all(y <= 0.5 + tolerance for y in y_coords)
            y_all_top = all(y >= 0.5 - tolerance for y in y_coords)
            
            if !(y_all_bottom || y_all_top)
                push!(violations_y, cell_idx)
            end
        end
        
        println("Cells violating x=0.5 constraint: $(length(violations_x))")
        println("Cells violating y=0.5 constraint: $(length(violations_y))")
        
        if isempty(violations_x) && isempty(violations_y)
            println("✓ All cells respect both constraints")
        else
            println("✗ Some cells violate constraints")
        end
        
        @test isempty(violations_x) && isempty(violations_y)
    end
end

println("\n" * "="^60)
println("CONSTRAINT ENFORCEMENT TEST SUMMARY")
println("="^60)
