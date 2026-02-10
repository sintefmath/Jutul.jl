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
    
    @testset "find_intersected_faces function" begin
        # Test finding faces intersected by a line segment
        # Create a simple 2x2 grid
        points = [0.25 0.75 0.25 0.75;
                  0.25 0.25 0.75 0.75]
        
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)
        
        @test mesh isa UnstructuredMesh
        @test number_of_cells(mesh) == 4
        
        # Find faces intersected by a vertical line at x=0.5
        intersections = Jutul.VoronoiMeshes.find_intersected_faces(mesh, [0.5, 0.0], [0.5, 1.0])
        
        # Should find at least one face
        @test length(intersections) > 0
        
        # Check that intersections are sorted by t parameter
        if length(intersections) > 1
            for i in 1:(length(intersections)-1)
                @test intersections[i].t <= intersections[i+1].t
            end
        end
    end
    
    @testset "insert_line_segment function" begin
        # Test that insert_line_segment actually splits cells
        Random.seed!(789)
        points = rand(2, 10)
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)
        
        original_ncells = number_of_cells(mesh)
        
        # Insert a vertical line segment
        mesh2 = Jutul.VoronoiMeshes.insert_line_segment(mesh, [0.5, 0.0], [0.5, 1.0])
        
        @test mesh2 isa UnstructuredMesh
        # Cell count should increase due to splitting
        @test number_of_cells(mesh2) > original_ncells
    end
    
    @testset "Cell splitting by single constraint" begin
        # Test that cells crossing a constraint are split
        points = [0.25 0.75 0.25 0.75;
                  0.25 0.25 0.75 0.75]
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)
        original_ncells = number_of_cells(mesh)
        
        # Insert horizontal constraint
        mesh2 = Jutul.VoronoiMeshes.insert_line_segment(mesh, [0.0, 0.5], [1.0, 0.5])
        
        @test number_of_cells(mesh2) > original_ncells
    end
    
    @testset "Cell splitting by multiple constraints" begin
        # Test multiple constraints
        Random.seed!(999)
        points = rand(2, 20)
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)
        original_ncells = number_of_cells(mesh)
        
        # Insert vertical constraint
        mesh2 = Jutul.VoronoiMeshes.insert_line_segment(mesh, [0.5, 0.2], [0.5, 0.8])
        mid_ncells = number_of_cells(mesh2)
        
        # Insert horizontal constraint
        mesh3 = Jutul.VoronoiMeshes.insert_line_segment(mesh2, [0.2, 0.5], [0.8, 0.5])
        final_ncells = number_of_cells(mesh3)
        
        @test mid_ncells > original_ncells
        @test final_ncells > mid_ncells
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
    
    @testset "Usage example from docstring" begin
        # Test the exact usage example from PEBIMesh2D docstring
        
        # Example 1: Simple mesh with 4 points
        points = [0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0]
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)
        @test mesh isa UnstructuredMesh
        @test number_of_cells(mesh) == 4
        
        # Example 2: Mesh with constraint line (use post-processing)
        Random.seed!(42)
        points = rand(2, 20)
        mesh = Jutul.VoronoiMeshes.PEBIMesh2D(points)
        @test number_of_cells(mesh) == 20
        
        # Add vertical constraint
        mesh = Jutul.VoronoiMeshes.insert_line_segment(mesh, [0.5, 0.0], [0.5, 1.0])
        
        # Result should have > 20 cells due to splitting
        @test number_of_cells(mesh) > 20
        @test mesh isa UnstructuredMesh
        
        # Verify mesh is still valid
        geo = tpfv_geometry(mesh)
        @test all(geo.volumes .> 0)  # All cells should have positive volume
    end
end
