using Test
using Jutul
using Random

@testset "PEBI/Voronoi Mesh Tests" begin
    @testset "Basic 2D PEBI mesh" begin
        # Test with 4 points in a square
        points = [0.0 1.0 0.0 1.0; 
                  0.0 0.0 1.0 1.0]
        
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)
        
        @test mesh isa UnstructuredMesh
        @test number_of_cells(mesh) == 4
        @test dim(mesh) == 2
        
        # Check that we have some faces
        @test number_of_faces(mesh) > 0
    end
    
    @testset "2D PEBI mesh with random points" begin
        # Test with random points
        Random.seed!(1234)
        points = rand(2, 10)
        
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)
        
        @test mesh isa UnstructuredMesh
        @test number_of_cells(mesh) == 10
        @test dim(mesh) == 2
    end
    
    @testset "2D PEBI mesh with custom bbox" begin
        # Test with custom bounding box
        points = [0.5 0.5; 0.3 0.7]
        bbox = ((0.0, 1.0), (0.0, 1.0))
        
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points, bbox=bbox)
        
        @test mesh isa UnstructuredMesh
        @test number_of_cells(mesh) == 2
    end
    
    @testset "2D PEBI mesh with constraints" begin
        # Test with a constraint line
        points = rand(2, 15)
        constraint = ([0.5, 0.0], [0.5, 1.0])  # Vertical line at x=0.5
        
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points, constraints=[constraint])
        
        @test mesh isa UnstructuredMesh
        # Should have at least the original 15 cells plus possibly cells for constraint points
        @test number_of_cells(mesh) >= 15
    end
    
    @testset "2D PEBI mesh constraint enforcement" begin
        # Test that constraints are actually enforced
        # Create points on both sides of a vertical line at x=0.5
        points = [0.2 0.8 0.2 0.8;
                  0.2 0.2 0.8 0.8]
        constraint = ([0.5, 0.0], [0.5, 1.0])  # Vertical line at x=0.5
        
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points, constraints=[constraint])
        
        @test mesh isa UnstructuredMesh
        
        # Get geometry
        geo = tpfv_geometry(mesh)
        
        # Check that no cell straddles the constraint line
        # All cell vertices should be on one side of x=0.5 or the other
        for cell_idx in 1:number_of_cells(mesh)
            # Get cell vertices by collecting vertices from all faces
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
            
            # Check x-coordinates of all vertices
            x_coords = [mesh.node_points[v][1] for v in cell_vertices]
            
            # All vertices should be on one side of 0.5, or exactly on 0.5
            tol = 1e-8
            all_left = all(x <= 0.5 + tol for x in x_coords)
            all_right = all(x >= 0.5 - tol for x in x_coords)
            
            # At least one should be true (cell doesn't straddle the constraint)
            @test all_left || all_right
        end
    end
    
    @testset "Cell splitting by single constraint" begin
        # Test that a cell crossing a constraint is split into two cells
        # Create a single point that will have a cell crossing the constraint
        points = [0.5; 0.5]'  # Single point at center (as 2×1 matrix)
        constraint = ([0.0, 0.5], [1.0, 0.5])  # Horizontal line through center
        
        # Mesh without constraint should have 1 cell
        mesh_no_constraint = Jutul.VoronoiMeshes.PEBIMesh2D(points)
        @test number_of_cells(mesh_no_constraint) == 1
        
        # Mesh with constraint should split the cell
        mesh_with_constraint = Jutul.VoronoiMeshes.PEBIMesh2D(points, constraints=[constraint])
        # The single cell should be split into 2 cells
        @test number_of_cells(mesh_with_constraint) >= 2
    end
    
    @testset "Cell splitting by multiple constraints" begin
        # Test that a cell can be split by multiple constraints
        # Single point with two perpendicular constraints
        points = [0.5; 0.5]'  # Single point at center (as 2×1 matrix)
        constraints = [
            ([0.0, 0.5], [1.0, 0.5]),  # Horizontal line
            ([0.5, 0.0], [0.5, 1.0])   # Vertical line
        ]
        
        # Mesh with two perpendicular constraints should split into 4 cells
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points, constraints=constraints)
        # The cell should be split into 4 quadrants
        @test number_of_cells(mesh) >= 4
    end
    
    @testset "Intersecting constraints with many points" begin
        # Test that intersecting constraints work correctly with many points
        # Solution 1: Pre-computed vertices should eliminate interior boundaries
        Random.seed!(42)
        points = rand(2, 100)
        constraints = [
            ([0.5, 0.2], [0.5, 0.8]),  # Vertical line
            ([0.2, 0.5], [0.8, 0.5])   # Horizontal line (forms cross)
        ]
        
        # With Solution 1, no warning should be generated
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points, constraints=constraints)
        
        # Should have more cells than original points due to splitting
        @test number_of_cells(mesh) > 100
        
        # Check for mesh validity - no holes
        geo = tpfv_geometry(mesh)
        @test all(geo.volumes .> 0)  # All cells should have positive volume
        
        # Verify no interior boundaries
        bbox_xmin, bbox_xmax = extrema([p[1] for p in mesh.node_points])
        bbox_ymin, bbox_ymax = extrema([p[2] for p in mesh.node_points])
        tol = 1e-6
        
        interior_boundary_count = 0
        for bface_idx in 1:number_of_boundary_faces(mesh)
            nodes = [mesh.boundary_faces.faces_to_nodes[bface_idx]...]
            all_on_boundary = true
            for node_idx in nodes
                pt = mesh.node_points[node_idx]
                on_boundary = (abs(pt[1] - bbox_xmin) < tol || abs(pt[1] - bbox_xmax) < tol ||
                              abs(pt[2] - bbox_ymin) < tol || abs(pt[2] - bbox_ymax) < tol)
                if !on_boundary
                    all_on_boundary = false
                    break
                end
            end
            if !all_on_boundary
                interior_boundary_count += 1
            end
        end
        
        # Solution 1 should fix interior boundaries
        @test interior_boundary_count == 0
    end
    
    @testset "2D PEBI mesh with high coordinate variation" begin
        # Test with points that have high coordinate variation
        points = [0.0 1000.0 500.0 250.0; 
                  0.0 0.0 1000.0 500.0]
        
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)
        
        @test mesh isa UnstructuredMesh
        @test number_of_cells(mesh) == 4
        
        # Verify geometry can be computed
        geo = tpfv_geometry(mesh)
        @test geo isa TwoPointFiniteVolumeGeometry
        @test length(geo.volumes) == 4
        @test all(geo.volumes .> 0)  # All volumes should be positive
    end
    
    @testset "3D PEBI mesh placeholder" begin
        # Test that 3D throws appropriate error
        points = rand(3, 8)
        
        @test_throws ErrorException Jutul.VoronoiMeshes.PEBIMesh3D(points)
    end
    
    @testset "Point format conversion" begin
        # Test different point input formats
        
        # Format 1: 2×n matrix
        points1 = [0.0 1.0; 0.0 1.0]
        mesh1 = Jutul.VoronoiMeshes.PEBIMesh2D(points1)
        @test number_of_cells(mesh1) == 2
        
        # Format 2: n×2 matrix (transposed)
        points2 = [0.0 0.0; 1.0 1.0]
        mesh2 = Jutul.VoronoiMeshes.PEBIMesh2D(points2)
        @test number_of_cells(mesh2) == 2
        
        # Format 3: Vector of tuples
        points3 = [(0.0, 0.0), (1.0, 1.0)]
        mesh3 = Jutul.VoronoiMeshes.PEBIMesh2D(points3)
        @test number_of_cells(mesh3) == 2
    end
end
