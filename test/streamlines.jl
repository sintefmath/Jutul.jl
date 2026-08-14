using Jutul
using Jutul.Streamlines
using Test
using LinearAlgebra
using StaticArrays

function constant_velocity_fluxes(mesh, geo, velocity::SVector{D, T}) where {D, T}
    nf = number_of_faces(mesh)
    fluxes = zeros(T, nf)
    for face in 1:nf
        normal = SVector{D, T}(geo.normals[:, face])
        fluxes[face] = geo.areas[face] * dot(velocity, normal)
    end
    return fluxes
end

function rotational_velocity_fluxes(mesh, geo)
    T = eltype(geo.areas)
    D = size(geo.cell_centroids, 1)
    mins = map(d -> minimum(pt[d] for pt in mesh.node_points), 1:D)
    maxs = map(d -> maximum(pt[d] for pt in mesh.node_points), 1:D)
    center = SVector{D, T}(ntuple(d -> T((mins[d] + maxs[d]) / 2), D))
    nf = number_of_faces(mesh)
    fluxes = zeros(T, nf)
    for face in 1:nf
        x = SVector{D, T}(geo.face_centroids[:, face])
        normal = SVector{D, T}(geo.normals[:, face])
        velocity = SVector{D, T}(-(x[2] - center[2]), x[1] - center[1])
        fluxes[face] = geo.areas[face] * dot(velocity, normal)
    end
    return fluxes
end

@testset "Streamline Tracing" begin
    @testset "2D Cartesian mesh streamline tracing" begin
        # Create a simple 2D Cartesian mesh
        dims = (5, 5)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        geo = tpfv_geometry(mesh)
        
        # Create a simple uniform flow field (constant velocity in x-direction)
        fluxes = constant_velocity_fluxes(mesh, geo, SVector(1.0, 0.0))
        
        # Setup streamline tracer
        tracer = setup_streamline_tracer(mesh, fluxes)
        
        # Test that tracer was created
        @test tracer isa Streamlines.StreamlineTracer
        @test length(tracer.subcells) > 0
        @test tracer.octree isa Streamlines.OctreeNode
        
        # Test streamline tracing from a single point
        start_point = SVector{2, Float64}(geo.cell_centroids[:, 1])
        streamlines = trace_streamlines(tracer, [start_point], max_steps = 100, step_size = 0.1)
        
        @test length(streamlines) == 1
        @test length(streamlines[1]) > 1
        @test last(streamlines[1])[1] > start_point[1]
        @test abs(last(streamlines[1])[2] - start_point[2]) < 1e-6
    end
    
    @testset "3D mesh sub-cell tesselation" begin
        # Create a simple 3D mesh
        dims = (3, 3, 3)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        # Create fluxes
        geo = tpfv_geometry(mesh)
        fluxes = constant_velocity_fluxes(mesh, geo, SVector(1.0, 0.25, -0.1))
        
        # Setup tracer
        tracer = setup_streamline_tracer(mesh, fluxes, max_depth = 6)
        
        # Verify sub-cells were created
        @test length(tracer.subcells) > 0
        
        # Each cell should produce multiple sub-cells (tetrahedra)
        nc = number_of_cells(mesh)
        avg_subcells_per_cell = length(tracer.subcells) / nc
        @test avg_subcells_per_cell > 1  # Should have more than 1 sub-cell per cell
        
        # Test that octree is non-empty
        @test tracer.octree isa Streamlines.OctreeNode
        @test tracer.octree.max_depth >= 0
    end
    
    @testset "Point location in octree" begin
        # Create a small 2D mesh
        dims = (4, 4)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        geo = tpfv_geometry(mesh)
        fluxes = constant_velocity_fluxes(mesh, geo, SVector(1.0, 0.0))

        tracer = setup_streamline_tracer(mesh, fluxes)
        
        # Test finding sub-cell at various cell centroids
        for cell in 1:min(5, number_of_cells(mesh))
            point = SVector{2, Float64}(geo.cell_centroids[:, cell])
            subcell_idx = Streamlines.find_subcell_at_point(tracer, point)
            @test !isnothing(subcell_idx)
            @test tracer.subcells[subcell_idx].parent_cell == cell
        end
    end

    @testset "Point location covers boundary cells" begin
        dims = (4, 4)
        mesh = UnstructuredMesh(CartesianMesh(dims))
        geo = tpfv_geometry(mesh)
        fluxes = constant_velocity_fluxes(mesh, geo, SVector(1.0, 0.0))
        tracer = setup_streamline_tracer(mesh, fluxes)

        for cell in 1:number_of_cells(mesh)
            cell_center = SVector{2, Float64}(geo.cell_centroids[:, cell])

            for face in mesh.faces.cells_to_faces[cell]
                face_center = SVector{2, Float64}(geo.face_centroids[:, face])
                pt = (cell_center + face_center) / 2
                idx = Streamlines.find_subcell_at_point(tracer, pt)
                @test !isnothing(idx)
                @test tracer.subcells[idx].parent_cell == cell
            end

            for face in mesh.boundary_faces.cells_to_faces[cell]
                face_center = SVector{2, Float64}(geo.boundary_centroids[:, face])
                pt = (cell_center + face_center) / 2
                idx = Streamlines.find_subcell_at_point(tracer, pt)
                @test !isnothing(idx)
                @test tracer.subcells[idx].parent_cell == cell
            end
        end
    end
    
    @testset "Forward and backward tracing" begin
        # Create a 2D mesh
        dims = (6, 6)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        geo = tpfv_geometry(mesh)
        fluxes = constant_velocity_fluxes(mesh, geo, SVector(1.0, 0.0))
        
        tracer = setup_streamline_tracer(mesh, fluxes)
        
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
        @test streamlines_both[1][1][1] <= start_point[1]
        @test streamlines_both[1][end][1] >= start_point[1]
    end
    
    @testset "Multiple starting points" begin
        # Create a mesh
        dims = (5, 5)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        geo = tpfv_geometry(mesh)
        fluxes = constant_velocity_fluxes(mesh, geo, SVector(1.0, 0.0))
        
        tracer = setup_streamline_tracer(mesh, fluxes)
        
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
        
        geo = tpfv_geometry(mesh)
        fluxes = constant_velocity_fluxes(mesh, geo, SVector(1.0, 0.0))
        
        tracer = setup_streamline_tracer(mesh, fluxes)
        
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
        
        @test Streamlines.bbox_overlap(point, bbox_min, bbox_max)
        
        # Point outside
        point_out = SVector{2, Float64}(1.5, 0.5)
        @test !Streamlines.bbox_overlap(point_out, bbox_min, bbox_max)
        
        # Point on boundary (should be inside with tolerance)
        point_boundary = SVector{2, Float64}(1.0, 1.0)
        @test Streamlines.bbox_overlap(point_boundary, bbox_min, bbox_max)
    end
    
    @testset "Tetrahedral centroid and volume" begin
        # Test tetrahedron calculations
        vertices = [
            SVector{3, Float64}(0.0, 0.0, 0.0),
            SVector{3, Float64}(1.0, 0.0, 0.0),
            SVector{3, Float64}(0.0, 1.0, 0.0),
            SVector{3, Float64}(0.0, 0.0, 1.0)
        ]
        
        centroid, volume = Streamlines.compute_tet_centroid_and_volume(vertices)
        
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
        
        centroid, area = Streamlines.compute_tri_centroid_and_area(vertices)
        
        # Centroid should be average of vertices
        expected_centroid = SVector{2, Float64}(1.0/3.0, 1.0/3.0)
        @test norm(centroid - expected_centroid) < 1e-10
        
        # Area of right triangle with legs 1
        @test abs(area - 0.5) < 1e-10
    end
    
    @testset "Integration methods" begin
        # Create a simple test case where we can compare different integrators
        dims = (5, 5)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        geo = tpfv_geometry(mesh)
        fluxes = rotational_velocity_fluxes(mesh, geo)
        
        tracer = setup_streamline_tracer(mesh, fluxes)
        
        # Pick a starting point
        start_point = SVector{2, Float64}(geo.cell_centroids[:, 1])
        
        # Test Euler integrator
        streamlines_euler = trace_streamlines(
            tracer, [start_point],
            max_steps = 100,
            step_size = 0.05,
            integrator = EulerIntegrator()
        )
        @test length(streamlines_euler) == 1
        @test length(streamlines_euler[1]) > 1
        
        # Test RK2 integrator
        streamlines_rk2 = trace_streamlines(
            tracer, [start_point],
            max_steps = 100,
            step_size = 0.05,
            integrator = RK2Integrator()
        )
        @test length(streamlines_rk2) == 1
        @test length(streamlines_rk2[1]) > 1
        
        # Test RK4 integrator
        streamlines_rk4 = trace_streamlines(
            tracer, [start_point],
            max_steps = 100,
            step_size = 0.05,
            integrator = RK4Integrator()
        )
        @test length(streamlines_rk4) == 1
        @test length(streamlines_rk4[1]) > 1
        
        # All methods should produce valid streamlines
        # They may differ slightly in path due to different accuracies
        # but all should start from the same point
        @test streamlines_euler[1][1] == start_point
        @test streamlines_rk2[1][1] == start_point
        @test streamlines_rk4[1][1] == start_point
        @test last(streamlines_euler[1]) != last(streamlines_rk4[1])
    end
    
    @testset "Default integrator" begin
        # Test that default integrator is Euler when not specified
        dims = (4, 4)
        g = CartesianMesh(dims)
        mesh = UnstructuredMesh(g)
        
        geo = tpfv_geometry(mesh)
        fluxes = constant_velocity_fluxes(mesh, geo, SVector(1.0, 0.0))
        
        tracer = setup_streamline_tracer(mesh, fluxes)
        
        start_point = SVector{2, Float64}(geo.cell_centroids[:, 1])
        
        # Call without specifying integrator (should use Euler by default)
        streamlines_default = trace_streamlines(
            tracer, [start_point],
            max_steps = 50
        )
        
        @test length(streamlines_default) == 1
        @test length(streamlines_default[1]) > 1
    end

    @testset "Different flux fields produce different paths" begin
        mesh = UnstructuredMesh(CartesianMesh((8, 8)))
        geo = tpfv_geometry(mesh)
        start_point = SVector{2, Float64}(geo.cell_centroids[:, 1 + 3*8])

        uniform_fluxes = constant_velocity_fluxes(mesh, geo, SVector(1.0, 0.0))
        rotational_fluxes = rotational_velocity_fluxes(mesh, geo)

        uniform_tracer = setup_streamline_tracer(mesh, uniform_fluxes)
        rotational_tracer = setup_streamline_tracer(mesh, rotational_fluxes)

        uniform_streamline = only(trace_streamlines(uniform_tracer, [start_point], max_steps = 80, step_size = 0.05))
        rotational_streamline = only(trace_streamlines(rotational_tracer, [start_point], max_steps = 80, step_size = 0.05))

        @test length(uniform_streamline) > 1
        @test length(rotational_streamline) > 1
        @test norm(last(uniform_streamline) - last(rotational_streamline)) > 0.1
    end
end
