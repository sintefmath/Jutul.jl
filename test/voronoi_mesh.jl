using Test
using Jutul

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
        using Random
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
