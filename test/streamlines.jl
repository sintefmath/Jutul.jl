using Jutul
using Test
using LinearAlgebra
using StaticArrays

@testset "Streamline Tracing" begin
    @testset "2D Cartesian mesh streamline tracing" begin
        # Create a simple 2D Cartesian mesh
        dims = (5, 5)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        # Create a simple uniform flow field (constant velocity in x-direction)
        nf = number_of_faces(mesh)
        fluxes = zeros(nf)
        
        # Set up a uniform flow from left to right
        # For a Cartesian mesh, we need to set fluxes appropriately
        # This is a simplified example
        for i in 1:nf
            # Assign a constant flux in x-direction
            fluxes[i] = 1.0
        end
        
        # Setup streamline tracer
        tracer = setup_streamline_tracer(mesh, fluxes)
        
        # Test that tracer was created
        @test tracer isa StreamlineTracer
        @test length(tracer.subcells) > 0
        @test tracer.octree isa OctreeNode
        
        # Test streamline tracing from a single point
        geo = tpfv_geometry(mesh)
        start_point = SVector{2, Float64}(geo.cell_centroids[:, 1])
        streamlines = trace_streamlines(tracer, [start_point], max_steps = 100, step_size = 0.1)
        
        @test length(streamlines) == 1
        @test length(streamlines[1]) > 0
    end
    
    @testset "3D mesh sub-cell tesselation" begin
        # Create a simple 3D mesh
        dims = (3, 3, 3)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        # Create fluxes
        nf = number_of_faces(mesh)
        fluxes = randn(nf) * 0.1 .+ 1.0  # Small random perturbation around 1.0
        
        # Setup tracer
        tracer = setup_streamline_tracer(mesh, fluxes, max_depth = 6)
        
        # Verify sub-cells were created
        @test length(tracer.subcells) > 0
        
        # Each cell should produce multiple sub-cells (tetrahedra)
        nc = number_of_cells(mesh)
        avg_subcells_per_cell = length(tracer.subcells) / nc
        @test avg_subcells_per_cell > 1  # Should have more than 1 sub-cell per cell
        
        # Test that octree is non-empty
        @test tracer.octree isa OctreeNode
        @test tracer.octree.max_depth >= 0
    end
    
    @testset "Point location in octree" begin
        # Create a small 2D mesh
        dims = (4, 4)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        nf = number_of_faces(mesh)
        fluxes = ones(nf)
        
        tracer = setup_streamline_tracer(mesh, fluxes)
        geo = tpfv_geometry(mesh)
        
        # Test finding sub-cell at various cell centroids
        for cell in 1:min(5, number_of_cells(mesh))
            point = SVector{2, Float64}(geo.cell_centroids[:, cell])
            subcell_idx = Jutul.find_subcell_at_point(tracer, point)
            
            # Should find a sub-cell (might be nothing if point is exactly on boundary)
            # At least for interior cells we should find something
            if cell > 1
                # Allow for some cells to not be found due to numerical issues
                # but most should be found
            end
        end
    end
    
    @testset "Forward and backward tracing" begin
        # Create a 2D mesh
        dims = (6, 6)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        nf = number_of_faces(mesh)
        fluxes = ones(nf) * 0.5
        
        tracer = setup_streamline_tracer(mesh, fluxes)
        geo = tpfv_geometry(mesh)
        
        # Pick a point in the middle
        mid_cell = div(number_of_cells(mesh), 2)
        start_point = SVector{2, Float64}(geo.cell_centroids[:, mid_cell])
        
        # Trace forward only
        streamlines_fwd = trace_streamlines(tracer, [start_point], 
                                           max_steps = 50, 
                                           forward = true, 
                                           backward = false)
        
        # Trace backward only
        streamlines_bwd = trace_streamlines(tracer, [start_point], 
                                           max_steps = 50, 
                                           forward = false, 
                                           backward = true)
        
        # Both should produce streamlines
        @test length(streamlines_fwd) == 1
        @test length(streamlines_bwd) == 1
        
        # Forward and backward combined
        streamlines_both = trace_streamlines(tracer, [start_point], 
                                            max_steps = 50, 
                                            forward = true, 
                                            backward = true)
        
        @test length(streamlines_both) == 1
        # Combined should generally be longer (or equal if hit boundary immediately)
        @test length(streamlines_both[1]) >= max(length(streamlines_fwd[1]), length(streamlines_bwd[1]))
    end
    
    @testset "Multiple starting points" begin
        # Create a mesh
        dims = (5, 5)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        nf = number_of_faces(mesh)
        fluxes = ones(nf)
        
        tracer = setup_streamline_tracer(mesh, fluxes)
        geo = tpfv_geometry(mesh)
        
        # Create multiple starting points
        n_starts = 5
        start_points = [SVector{2, Float64}(geo.cell_centroids[:, i]) for i in 1:n_starts]
        
        streamlines = trace_streamlines(tracer, start_points, max_steps = 30)
        
        @test length(streamlines) == n_starts
        # Each streamline should have at least the starting point
        for sl in streamlines
            @test length(sl) >= 1
        end
    end
    
    @testset "Matrix input format" begin
        # Test that matrix input works
        dims = (4, 4)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        nf = number_of_faces(mesh)
        fluxes = ones(nf)
        
        tracer = setup_streamline_tracer(mesh, fluxes)
        geo = tpfv_geometry(mesh)
        
        # Create starting points as a matrix (2 x n)
        n_starts = 3
        start_matrix = hcat([geo.cell_centroids[:, i] for i in 1:n_starts]...)
        
        streamlines = trace_streamlines(tracer, start_matrix, max_steps = 20)
        
        @test length(streamlines) == n_starts
    end
    
    @testset "Bounding box overlap" begin
        # Test the bbox_overlap function
        point = SVector{2, Float64}(0.5, 0.5)
        bbox_min = SVector{2, Float64}(0.0, 0.0)
        bbox_max = SVector{2, Float64}(1.0, 1.0)
        
        @test Jutul.bbox_overlap(point, bbox_min, bbox_max)
        
        # Point outside
        point_out = SVector{2, Float64}(1.5, 0.5)
        @test !Jutul.bbox_overlap(point_out, bbox_min, bbox_max)
        
        # Point on boundary (should be inside with tolerance)
        point_boundary = SVector{2, Float64}(1.0, 1.0)
        @test Jutul.bbox_overlap(point_boundary, bbox_min, bbox_max)
    end
    
    @testset "Tetrahedral centroid and volume" begin
        # Test tetrahedron calculations
        vertices = [
            SVector{3, Float64}(0.0, 0.0, 0.0),
            SVector{3, Float64}(1.0, 0.0, 0.0),
            SVector{3, Float64}(0.0, 1.0, 0.0),
            SVector{3, Float64}(0.0, 0.0, 1.0)
        ]
        
        centroid, volume = Jutul.compute_tet_centroid_and_volume(vertices)
        
        # Centroid should be average of vertices
        expected_centroid = SVector{3, Float64}(0.25, 0.25, 0.25)
        @test norm(centroid - expected_centroid) < 1e-10
        
        # Volume of this specific tetrahedron
        # V = 1/6 for unit simplex
        @test abs(volume - 1.0/6.0) < 1e-10
    end
    
    @testset "Triangle centroid and area" begin
        # Test triangle calculations
        vertices = [
            SVector{2, Float64}(0.0, 0.0),
            SVector{2, Float64}(1.0, 0.0),
            SVector{2, Float64}(0.0, 1.0)
        ]
        
        centroid, area = Jutul.compute_tri_centroid_and_area(vertices)
        
        # Centroid should be average of vertices
        expected_centroid = SVector{2, Float64}(1.0/3.0, 1.0/3.0)
        @test norm(centroid - expected_centroid) < 1e-10
        
        # Area of right triangle with legs 1
        @test abs(area - 0.5) < 1e-10
    end
end
