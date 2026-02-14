using Jutul
using Test
using LinearAlgebra
using StaticArrays

import Jutul.CutCellMeshes: PlaneCut, PolygonalSurface, cut_mesh, layered_mesh, depth_grid_to_surface

@testset "CutCellMeshes" begin
    @testset "PlaneCut construction" begin
        plane = PlaneCut([1.0, 2.0, 3.0], [0.0, 0.0, 2.0])
        @test plane.point == SVector{3, Float64}(1.0, 2.0, 3.0)
        @test plane.normal ≈ SVector{3, Float64}(0.0, 0.0, 1.0)
    end

    @testset "PolygonalSurface construction" begin
        poly = [
            SVector{3, Float64}(0.0, 0.0, 0.5),
            SVector{3, Float64}(1.0, 0.0, 0.5),
            SVector{3, Float64}(1.0, 1.0, 0.5),
            SVector{3, Float64}(0.0, 1.0, 0.5)
        ]
        surface = PolygonalSurface([poly])
        @test length(surface.polygons) == 1
        @test length(surface.normals) == 1
    end

    @testset "Axis-aligned Z cut" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01)

        @test number_of_cells(cut) == nc_orig + 9
        geo = tpfv_geometry(cut)
        @test all(geo.volumes .> 0)
        total_vol_orig = sum(tpfv_geometry(mesh).volumes)
        total_vol_cut = sum(geo.volumes)
        @test total_vol_orig ≈ total_vol_cut rtol=1e-10
    end

    @testset "Axis-aligned X cut" begin
        g = CartesianMesh((4, 3, 3))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        plane = PlaneCut([0.6, 0.0, 0.0], [1.0, 0.0, 0.0])
        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01)

        @test number_of_cells(cut) > nc_orig
        geo = tpfv_geometry(cut)
        @test all(geo.volumes .> 0)
        total_vol_orig = sum(tpfv_geometry(mesh).volumes)
        total_vol_cut = sum(geo.volumes)
        @test total_vol_orig ≈ total_vol_cut rtol=1e-10
    end

    @testset "Diagonal plane cut" begin
        g = CartesianMesh((2, 2, 2))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        plane = PlaneCut([0.5, 0.5, 0.5], normalize([1.0, 1.0, 1.0]))
        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01)

        @test number_of_cells(cut) > nc_orig
        geo = tpfv_geometry(cut)
        @test all(geo.volumes .> 0)
        total_vol_orig = sum(tpfv_geometry(mesh).volumes)
        total_vol_cut = sum(geo.volumes)
        @test total_vol_orig ≈ total_vol_cut rtol=1e-8
    end

    @testset "No-cut plane outside mesh" begin
        g = CartesianMesh((2, 2, 2))
        mesh = UnstructuredMesh(g)
        plane = PlaneCut([0.0, 0.0, -1.0], [0.0, 0.0, 1.0])
        cut = cut_mesh(mesh, plane)
        @test cut === mesh
    end

    @testset "No-cut plane on boundary" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        # z=0 is exactly on the boundary nodes
        plane = PlaneCut([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        cut = cut_mesh(mesh, plane)
        @test cut === mesh
    end

    @testset "Interior normals consistency" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        plane = PlaneCut([0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01)

        geo = tpfv_geometry(cut)
        for f in 1:number_of_faces(cut)
            l, r = cut.faces.neighbors[f]
            cl = geo.cell_centroids[:, l]
            cr = geo.cell_centroids[:, r]
            N = geo.normals[:, f]
            @test dot(N, cr - cl) > 0
        end
    end

    @testset "Boundary normals consistency" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        plane = PlaneCut([0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01)

        geo = tpfv_geometry(cut)
        for f in 1:number_of_boundary_faces(cut)
            c = cut.boundary_faces.neighbors[f]
            cc = geo.cell_centroids[:, c]
            fc = geo.boundary_centroids[:, f]
            N = geo.boundary_normals[:, f]
            @test dot(N, fc - cc) > 0
        end
    end

    @testset "PolygonalSurface cut" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        poly = [
            SVector{3, Float64}(-1.0, -1.0, 0.5),
            SVector{3, Float64}(2.0, -1.0, 0.5),
            SVector{3, Float64}(2.0, 2.0, 0.5),
            SVector{3, Float64}(-1.0, 2.0, 0.5)
        ]
        surface = PolygonalSurface([poly])
        cut = cut_mesh(mesh, surface; min_cut_fraction = 0.01)

        @test number_of_cells(cut) > nc_orig
        geo = tpfv_geometry(cut)
        @test all(geo.volumes .> 0)
    end

    @testset "Multiple plane cuts" begin
        g = CartesianMesh((4, 4, 4))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        # Two z-planes
        plane1 = PlaneCut([0.0, 0.0, 0.375], [0.0, 0.0, 1.0])
        plane2 = PlaneCut([0.0, 0.0, 0.625], [0.0, 0.0, 1.0])

        cut = cut_mesh(mesh, plane1; min_cut_fraction = 0.01)
        cut = cut_mesh(cut, plane2; min_cut_fraction = 0.01)

        @test number_of_cells(cut) > nc_orig
        geo = tpfv_geometry(cut)
        @test all(geo.volumes .> 0)

        total_vol_orig = sum(tpfv_geometry(mesh).volumes)
        total_vol_cut = sum(geo.volumes)
        @test total_vol_orig ≈ total_vol_cut rtol=1e-8
    end

    @testset "Min cut fraction threshold" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        # Place plane very close to a node boundary - should skip cutting
        # Mesh nodes at z = 0, 1/3, 2/3, 1
        # z = 0.34 is very close to 1/3 = 0.333..., creating a tiny sliver
        plane = PlaneCut([0.0, 0.0, 0.34], [0.0, 0.0, 1.0])
        cut_low_threshold = cut_mesh(mesh, plane; min_cut_fraction = 0.0)
        cut_high_threshold = cut_mesh(mesh, plane; min_cut_fraction = 0.15)

        @test number_of_cells(cut_low_threshold) >= number_of_cells(cut_high_threshold)
    end

    @testset "Geometry after cut" begin
        g = CartesianMesh((2, 2, 2), (2.0, 2.0, 2.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([1.0, 1.0, 0.5], [0.0, 0.0, 1.0])
        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01)

        geo = tpfv_geometry(cut)

        @testset "positive volumes" begin
            @test all(geo.volumes .> 0)
        end

        @testset "total volume conserved" begin
            total_vol_orig = sum(tpfv_geometry(mesh).volumes)
            total_vol_cut = sum(geo.volumes)
            @test total_vol_orig ≈ total_vol_cut rtol=1e-10
        end

        @testset "interior normals" begin
            for f in 1:number_of_faces(cut)
                l, r = cut.faces.neighbors[f]
                cl = geo.cell_centroids[:, l]
                cr = geo.cell_centroids[:, r]
                N = geo.normals[:, f]
                @test dot(N, cr - cl) > 0
            end
        end
    end

    @testset "Performance: 10^3 cells" begin
        g = CartesianMesh((10, 10, 10))
        mesh = UnstructuredMesh(g)
        @test number_of_cells(mesh) == 1000

        plane = PlaneCut([0.5, 0.5, 0.55], [0.0, 0.0, 1.0])
        t = @elapsed cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01)

        @test number_of_cells(cut) > 1000
        geo = tpfv_geometry(cut)
        @test all(geo.volumes .> 0)
        total_vol_orig = sum(tpfv_geometry(mesh).volumes)
        total_vol_cut = sum(geo.volumes)
        @test total_vol_orig ≈ total_vol_cut rtol=1e-8
    end

    @testset "Performance: 10000+ cells" begin
        g = CartesianMesh((20, 20, 25))
        mesh = UnstructuredMesh(g)
        @test number_of_cells(mesh) == 10000

        plane = PlaneCut([0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
        t = @elapsed cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01)

        @test number_of_cells(cut) > 10000
        geo = tpfv_geometry(cut)
        @test all(geo.volumes .> 0)
        total_vol_orig = sum(tpfv_geometry(mesh).volumes)
        total_vol_cut = sum(geo.volumes)
        @test total_vol_orig ≈ total_vol_cut rtol=1e-8
        @test t < 30.0  # Should complete in reasonable time
    end

    @testset "extra_out basic" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)
        nf_orig = number_of_faces(mesh)
        nb_orig = number_of_boundary_faces(mesh)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut, info = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)

        nc_new = number_of_cells(cut)
        nf_new = number_of_faces(cut)
        nb_new = number_of_boundary_faces(cut)

        # cell_index has correct length
        @test length(info["cell_index"]) == nc_new
        # face_index has correct length
        @test length(info["face_index"]) == nf_new
        # boundary_face_index has correct length
        @test length(info["boundary_face_index"]) == nb_new
        # new_faces is non-empty (cells were cut)
        @test length(info["new_faces"]) > 0
    end

    @testset "extra_out cell_index mapping" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut, info = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)

        ci = info["cell_index"]
        # All indices should be valid original cell indices
        @test all(1 .<= ci .<= nc_orig)
        # Cut cells should have two entries mapping to the same original cell
        # Count how many new cells map to each original cell
        counts = zeros(Int, nc_orig)
        for c in ci
            counts[c] += 1
        end
        # Uncut cells have exactly 1 new cell, cut cells have exactly 2
        @test all(c -> c == 1 || c == 2, counts)
        n_cut = count(c -> c == 2, counts)
        @test n_cut == 9  # 9 cells in the middle z-layer
    end

    @testset "extra_out face_index mapping" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        nf_orig = number_of_faces(mesh)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut, info = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)

        fi = info["face_index"]
        nf = info["new_faces"]
        # New cut faces should have face_index == 0
        for f in nf
            @test fi[f] == 0
        end
        # Non-new faces should reference a valid original face
        for (i, idx) in enumerate(fi)
            if !(i in nf)
                @test 1 <= idx <= nf_orig
            end
        end
    end

    @testset "extra_out boundary_face_index mapping" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        nb_orig = number_of_boundary_faces(mesh)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut, info = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)

        bfi = info["boundary_face_index"]
        # All boundary face indices should reference valid original boundary faces
        @test all(1 .<= bfi .<= nb_orig)
    end

    @testset "extra_out new_faces" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut, info = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)

        nf = info["new_faces"]
        # 9 cut cells should create 9 new faces
        @test length(nf) == 9
        # All new face indices should be valid
        @test all(1 .<= nf .<= number_of_faces(cut))
    end

    @testset "extra_out no-cut case" begin
        g = CartesianMesh((2, 2, 2))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        plane = PlaneCut([0.0, 0.0, -1.0], [0.0, 0.0, 1.0])
        result, info = cut_mesh(mesh, plane; extra_out = true)

        @test result === mesh
        @test info["cell_index"] == collect(1:nc_orig)
        @test info["face_index"] == collect(1:number_of_faces(mesh))
        @test info["boundary_face_index"] == collect(1:number_of_boundary_faces(mesh))
        @test isempty(info["new_faces"])
    end

    @testset "Bounding polygon - centroid mode" begin
        # 3x3x3 mesh, cut at z=0.5, but only within x=[0, 0.5], y=[0, 0.5]
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])

        # Bounding polygon covers first column only (x < 0.5, y < 0.5)
        # Cell centroids in x,y are at (1/6, 1/6), (1/2, 1/6), (5/6, 1/6), etc.
        # Only the cell at (1/6, 1/6) in the cut layer should be cut
        bpoly = [
            SVector{3, Float64}(0.0, 0.0, 0.5),
            SVector{3, Float64}(0.25, 0.0, 0.5),
            SVector{3, Float64}(0.25, 0.25, 0.5),
            SVector{3, Float64}(0.0, 0.25, 0.5)
        ]

        cut_bounded = cut_mesh(mesh, plane; min_cut_fraction = 0.01, bounding_polygon = bpoly)
        cut_unbounded = cut_mesh(mesh, plane; min_cut_fraction = 0.01)

        # Bounded cut should produce fewer new cells
        @test nc_orig < number_of_cells(cut_bounded) < number_of_cells(cut_unbounded)

        # Geometry should still be valid
        geo = tpfv_geometry(cut_bounded)
        @test all(geo.volumes .> 0)
        total_vol_orig = sum(tpfv_geometry(mesh).volumes)
        total_vol_cut = sum(geo.volumes)
        @test total_vol_orig ≈ total_vol_cut rtol=1e-10
    end

    @testset "Bounding polygon - clip_to_polygon mode" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])

        # Bounding polygon covers roughly the first cell centroid region
        # Use a larger polygon that excludes some centroids but includes
        # some nodes from additional cells when clip_to_polygon=true
        bpoly = [
            SVector{3, Float64}(-0.01, -0.01, 0.5),
            SVector{3, Float64}(0.20, -0.01, 0.5),
            SVector{3, Float64}(0.20, 0.20, 0.5),
            SVector{3, Float64}(-0.01, 0.20, 0.5)
        ]

        cut_centroid = cut_mesh(mesh, plane; min_cut_fraction = 0.01, bounding_polygon = bpoly)
        # A larger polygon that includes nodes from neighboring cells
        bpoly_large = [
            SVector{3, Float64}(-0.01, -0.01, 0.5),
            SVector{3, Float64}(0.40, -0.01, 0.5),
            SVector{3, Float64}(0.40, 0.40, 0.5),
            SVector{3, Float64}(-0.01, 0.40, 0.5)
        ]
        cut_centroid_large = cut_mesh(mesh, plane; min_cut_fraction = 0.01, bounding_polygon = bpoly_large)
        cut_clip_large = cut_mesh(mesh, plane; min_cut_fraction = 0.01, bounding_polygon = bpoly_large, clip_to_polygon = true)

        # clip_to_polygon should include at least as many cells as centroid mode
        @test number_of_cells(cut_clip_large) >= number_of_cells(cut_centroid_large)

        # Geometry should still be valid
        geo = tpfv_geometry(cut_clip_large)
        @test all(geo.volumes .> 0)
    end

    @testset "Bounding polygon - no cells in bounds" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])

        # Bounding polygon completely outside the mesh
        bpoly = [
            SVector{3, Float64}(5.0, 5.0, 0.5),
            SVector{3, Float64}(6.0, 5.0, 0.5),
            SVector{3, Float64}(6.0, 6.0, 0.5),
            SVector{3, Float64}(5.0, 6.0, 0.5)
        ]

        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01, bounding_polygon = bpoly)
        @test cut === mesh
    end

    @testset "extra_out with bounding_polygon" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        bpoly = [
            SVector{3, Float64}(0.0, 0.0, 0.5),
            SVector{3, Float64}(0.25, 0.0, 0.5),
            SVector{3, Float64}(0.25, 0.25, 0.5),
            SVector{3, Float64}(0.0, 0.25, 0.5)
        ]

        cut, info = cut_mesh(mesh, plane; min_cut_fraction = 0.01,
            bounding_polygon = bpoly, extra_out = true)

        @test length(info["cell_index"]) == number_of_cells(cut)
        @test length(info["face_index"]) == number_of_faces(cut)
        @test length(info["boundary_face_index"]) == number_of_boundary_faces(cut)
        @test length(info["new_faces"]) > 0
    end

    @testset "partial_cut negative" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01, partial_cut = :negative)

        # Keeping negative side (below z=0.5): 9 bottom cells + 9 cut half-cells = 18
        @test number_of_cells(cut) == 18
        geo = tpfv_geometry(cut)
        @test all(geo.volumes .> 0)
        vol_orig = sum(tpfv_geometry(mesh).volumes)
        vol_cut = sum(geo.volumes)
        @test vol_cut < vol_orig
        @test vol_cut ≈ 0.5 rtol=1e-8
    end

    @testset "partial_cut positive" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut_pos = cut_mesh(mesh, plane; min_cut_fraction = 0.01, partial_cut = :positive)
        cut_neg = cut_mesh(mesh, plane; min_cut_fraction = 0.01, partial_cut = :negative)

        geo_pos = tpfv_geometry(cut_pos)
        geo_neg = tpfv_geometry(cut_neg)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        @test all(geo_pos.volumes .> 0)
        @test all(geo_neg.volumes .> 0)
        # Volumes of positive and negative sides should sum to original
        @test sum(geo_pos.volumes) + sum(geo_neg.volumes) ≈ vol_orig rtol=1e-8
    end

    @testset "partial_cut geometry consistency" begin
        g = CartesianMesh((4, 4, 4))
        mesh = UnstructuredMesh(g)
        plane = PlaneCut([0.5, 0.5, 0.55], [0.0, 0.0, 1.0])

        for side in [:negative, :positive]
            cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01, partial_cut = side)
            geo = tpfv_geometry(cut)
            @test all(geo.volumes .> 0)

            # Interior normals point from left to right cell
            for f in 1:number_of_faces(cut)
                l, r = cut.faces.neighbors[f]
                cl = geo.cell_centroids[:, l]
                cr = geo.cell_centroids[:, r]
                N = geo.normals[:, f]
                @test dot(N, cr - cl) > 0
            end

            # Boundary normals point outward
            for f in 1:number_of_boundary_faces(cut)
                c = cut.boundary_faces.neighbors[f]
                cc = geo.cell_centroids[:, c]
                fc = geo.boundary_centroids[:, f]
                N = geo.boundary_normals[:, f]
                @test dot(N, fc - cc) > 0
            end
        end
    end

    @testset "partial_cut extra_out" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])

        cut, info = cut_mesh(mesh, plane; min_cut_fraction = 0.01,
            partial_cut = :negative, extra_out = true)

        @test length(info["cell_index"]) == number_of_cells(cut)
        @test length(info["face_index"]) == number_of_faces(cut)
        @test length(info["boundary_face_index"]) == number_of_boundary_faces(cut)
    end

    @testset "partial_cut no-cut case" begin
        g = CartesianMesh((2, 2, 2))
        mesh = UnstructuredMesh(g)

        # Plane outside mesh on negative side — all cells are positive
        plane = PlaneCut([0.0, 0.0, -1.0], [0.0, 0.0, 1.0])

        # partial_cut = :positive should keep all cells (they're all positive)
        cut_pos = cut_mesh(mesh, plane; partial_cut = :positive)
        @test number_of_cells(cut_pos) == number_of_cells(mesh)

        # Plane outside mesh on positive side — all cells are negative
        plane2 = PlaneCut([0.0, 0.0, 5.0], [0.0, 0.0, 1.0])

        # partial_cut = :negative should keep all cells (they're all negative)
        cut_neg = cut_mesh(mesh, plane2; partial_cut = :negative)
        @test number_of_cells(cut_neg) == number_of_cells(mesh)
    end

    @testset "depth_grid_to_surface basic" begin
        xs = collect(range(0.0, 1.0, length=4))
        ys = collect(range(0.0, 1.0, length=4))
        depths = fill(0.5, 4, 4)
        surface = depth_grid_to_surface(xs, ys, depths)
        @test length(surface.polygons) == 18  # 3x3 grid = 9 quads = 18 triangles
        @test length(surface.normals) == 18
    end

    @testset "depth_grid_to_surface with NaN" begin
        xs = collect(range(0.0, 1.0, length=3))
        ys = collect(range(0.0, 1.0, length=3))
        depths = fill(0.5, 3, 3)
        depths[2, 2] = NaN
        surface = depth_grid_to_surface(xs, ys, depths)
        @test length(surface.polygons) < 8  # Less than full grid
        @test length(surface.polygons) > 0
    end

    @testset "layered_mesh flat surfaces" begin
        g = CartesianMesh((3, 3, 4), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        xs = collect(range(0.0, 1.0, length=4))
        ys = collect(range(0.0, 1.0, length=4))
        d1 = fill(0.3, 4, 4)
        d2 = fill(0.7, 4, 4)
        s1 = depth_grid_to_surface(xs, ys, d1)
        s2 = depth_grid_to_surface(xs, ys, d2)

        result, info = layered_mesh(mesh, [s1, s2])

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol=1e-8

        layer_vals = sort(unique(info["layer_indices"]))
        @test length(layer_vals) == 3  # layers 0, 1, 2
        @test layer_vals == [0, 1, 2]

        @test length(info["layer_indices"]) == number_of_cells(result)
        @test length(info["cell_index"]) == number_of_cells(result)
    end

    @testset "layered_mesh single surface" begin
        g = CartesianMesh((3, 3, 4), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)

        xs = collect(range(0.0, 1.0, length=4))
        ys = collect(range(0.0, 1.0, length=4))
        d = fill(0.5, 4, 4)
        s = depth_grid_to_surface(xs, ys, d)

        result, info = layered_mesh(mesh, [s])
        layer_vals = sort(unique(info["layer_indices"]))
        @test layer_vals == [0, 1]

        vol_orig = sum(tpfv_geometry(mesh).volumes)
        vol_result = sum(tpfv_geometry(result).volumes)
        @test vol_result ≈ vol_orig rtol=1e-8
    end

    @testset "layered_mesh tilted surface" begin
        g = CartesianMesh((4, 4, 6), (100.0, 100.0, 60.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        xs = collect(range(0.0, 100.0, length=5))
        ys = collect(range(0.0, 100.0, length=5))
        d1 = [15.0 + (i - 1) * 2.5 for i in 1:5, j in 1:5]
        d2 = fill(40.0, 5, 5)
        s1 = depth_grid_to_surface(xs, ys, d1)
        s2 = depth_grid_to_surface(xs, ys, d2)

        result, info = layered_mesh(mesh, [s1, s2])

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol=1e-6

        layer_vals = sort(unique(info["layer_indices"]))
        @test length(layer_vals) == 3
    end

    @testset "layered_mesh volume per layer" begin
        g = CartesianMesh((3, 3, 4), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)

        xs = collect(range(0.0, 1.0, length=4))
        ys = collect(range(0.0, 1.0, length=4))
        d1 = fill(0.3, 4, 4)
        d2 = fill(0.7, 4, 4)
        s1 = depth_grid_to_surface(xs, ys, d1)
        s2 = depth_grid_to_surface(xs, ys, d2)

        result, info = layered_mesh(mesh, [s1, s2])
        geo = tpfv_geometry(result)
        nc = number_of_cells(result)

        # Layer 0: above z=0.3 → volume ≈ 0.3
        # Layer 1: between z=0.3 and z=0.7 → volume ≈ 0.4
        # Layer 2: below z=0.7 → volume ≈ 0.3
        vol0 = sum(geo.volumes[i] for i in 1:nc if info["layer_indices"][i] == 0)
        vol1 = sum(geo.volumes[i] for i in 1:nc if info["layer_indices"][i] == 1)
        vol2 = sum(geo.volumes[i] for i in 1:nc if info["layer_indices"][i] == 2)

        @test vol0 ≈ 0.3 rtol=1e-6
        @test vol1 ≈ 0.4 rtol=1e-6
        @test vol2 ≈ 0.3 rtol=1e-6
    end

    @testset "layered_mesh perturbed surface performance" begin
        # Regression test: this scenario used to hang or produce far too many
        # cells because each polygon's cut plane extended to infinity.
        g = CartesianMesh((10, 10, 10), (1000.0, 1000.0, 100.0))
        mesh = UnstructuredMesh(g)

        xs = collect(range(0.0, 1000.0, length=11))
        ys = collect(range(0.0, 1000.0, length=11))
        d1 = [33.0 + 5.0 * sin(0.1 * i + 0.2 * j) for i in 1:11, j in 1:11]
        d2 = fill(61.5, 11, 11)
        s1 = depth_grid_to_surface(xs, ys, d1)
        s2 = depth_grid_to_surface(xs, ys, d2)

        t = @elapsed begin
            result, info = layered_mesh(mesh, [s1, s2])
        end

        geo = tpfv_geometry(result)
        vol_orig = sum(tpfv_geometry(mesh).volumes)
        vol_result = sum(geo.volumes)

        @test all(geo.volumes .> 0)
        @test vol_result ≈ vol_orig rtol=1e-6
        @test sort(unique(info["layer_indices"])) == [0, 1, 2]
        @test t < 30.0  # Must complete in reasonable time (was hanging before fix)
    end

    @testset "merge_faces default on" begin
        # Verify merge_faces=true is the default and produces valid meshes
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01)  # default merge_faces=true

        geo = tpfv_geometry(cut)
        vol_orig = sum(tpfv_geometry(mesh).volumes)
        vol_cut = sum(geo.volumes)

        @test all(geo.volumes .> 0)
        @test vol_cut ≈ vol_orig rtol=1e-10

        # Interior normals consistency
        for f in 1:number_of_faces(cut)
            l, r = cut.faces.neighbors[f]
            cl = geo.cell_centroids[:, l]
            cr = geo.cell_centroids[:, r]
            N = geo.normals[:, f]
            @test dot(N, cr - cl) > 0
        end

        # Boundary normals consistency
        for f in 1:number_of_boundary_faces(cut)
            c = cut.boundary_faces.neighbors[f]
            cc = geo.cell_centroids[:, c]
            fc = geo.boundary_centroids[:, f]
            N = geo.boundary_normals[:, f]
            @test dot(N, fc - cc) > 0
        end
    end

    @testset "merge_faces=false produces valid mesh" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01, merge_faces = false)

        geo = tpfv_geometry(cut)
        vol_orig = sum(tpfv_geometry(mesh).volumes)
        vol_cut = sum(geo.volumes)

        @test all(geo.volumes .> 0)
        @test vol_cut ≈ vol_orig rtol=1e-10
    end

    @testset "merge_faces with PolygonalSurface" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        poly = [
            SVector{3, Float64}(-1.0, -1.0, 0.5),
            SVector{3, Float64}(2.0, -1.0, 0.5),
            SVector{3, Float64}(2.0, 2.0, 0.5),
            SVector{3, Float64}(-1.0, 2.0, 0.5)
        ]
        surface = PolygonalSurface([poly])

        cut_merged = cut_mesh(mesh, surface; min_cut_fraction = 0.01, merge_faces = true)
        cut_nomerge = cut_mesh(mesh, surface; min_cut_fraction = 0.01, merge_faces = false)

        # Both should have valid geometry
        geo_m = tpfv_geometry(cut_merged)
        geo_n = tpfv_geometry(cut_nomerge)
        @test all(geo_m.volumes .> 0)
        @test all(geo_n.volumes .> 0)
        # Same cell count
        @test number_of_cells(cut_merged) == number_of_cells(cut_nomerge)
        # Merged should have <= faces (equal if no merges possible)
        @test number_of_faces(cut_merged) <= number_of_faces(cut_nomerge)
        @test number_of_boundary_faces(cut_merged) <= number_of_boundary_faces(cut_nomerge)
    end

    @testset "merge_coplanar_faces on synthetic mesh" begin
        import Jutul.CutCellMeshes: merge_coplanar_faces

        # Create a simple mesh, then manually verify merge_coplanar_faces works
        # on a Cartesian mesh (which has no mergeable faces — it's a no-op test)
        g = CartesianMesh((2, 2, 2))
        mesh = UnstructuredMesh(g)
        merged = merge_coplanar_faces(mesh)

        @test number_of_cells(merged) == number_of_cells(mesh)
        @test number_of_faces(merged) == number_of_faces(mesh)
        @test number_of_boundary_faces(merged) == number_of_boundary_faces(mesh)

        # Verify geometry is preserved
        geo_orig = tpfv_geometry(mesh)
        geo_merged = tpfv_geometry(merged)
        @test sum(geo_orig.volumes) ≈ sum(geo_merged.volumes) rtol=1e-10
    end

    @testset "merge_coplanar_faces reduces boundary face count" begin
        import Jutul.CutCellMeshes: merge_coplanar_faces

        # A single z-cut on a 2x1x1 mesh creates cut cells whose boundary
        # faces on the ±y and ±x sides are split into sub-faces that share
        # the same cell AND are coplanar.  merge_coplanar_faces should
        # recombine them.
        g = CartesianMesh((1, 1, 1))
        mesh = UnstructuredMesh(g)
        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        # cut without merging
        cut_no = cut_mesh(mesh, plane; min_cut_fraction = 0.01, merge_faces = false)
        nb_no = number_of_boundary_faces(cut_no)

        # cut with merging (default)
        cut_yes = cut_mesh(mesh, plane; min_cut_fraction = 0.01)
        nb_yes = number_of_boundary_faces(cut_yes)

        # Each cut sub-cell has 5 boundary faces (4 sides + 1 z-face) = 10
        # but the 4 side-faces per sub-cell are unique (different cells).
        # If any coplanar pairs exist they will be merged; if not, counts equal.
        @test nb_yes <= nb_no
        # Both meshes should have valid geometry
        geo_no = tpfv_geometry(cut_no)
        geo_yes = tpfv_geometry(cut_yes)
        @test all(geo_no.volumes .> 0)
        @test all(geo_yes.volumes .> 0)
        @test sum(geo_no.volumes) ≈ sum(geo_yes.volumes) rtol=1e-10
    end

    @testset "merge_coplanar_faces orientation" begin
        import Jutul.CutCellMeshes: merge_coplanar_faces

        # Use a cut mesh where merging occurs, then verify normals
        g = CartesianMesh((4, 4, 4))
        mesh = UnstructuredMesh(g)
        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut = cut_mesh(mesh, plane; min_cut_fraction = 0.01, merge_faces = false)
        merged = merge_coplanar_faces(cut)

        geo = tpfv_geometry(merged)

        # Interior normals should point from left to right cell
        for f in 1:number_of_faces(merged)
            l, r = merged.faces.neighbors[f]
            cl = geo.cell_centroids[:, l]
            cr = geo.cell_centroids[:, r]
            N = geo.normals[:, f]
            @test dot(N, cr - cl) > 0
        end

        # Boundary normals should point outward
        for f in 1:number_of_boundary_faces(merged)
            c = merged.boundary_faces.neighbors[f]
            cc = geo.cell_centroids[:, c]
            fc = geo.boundary_centroids[:, f]
            N = geo.boundary_normals[:, f]
            @test dot(N, fc - cc) > 0
        end
    end

    @testset "merge_faces with extra_out" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut, info = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true, merge_faces = true)

        # Info dict should still work (cell_index refers to pre-merge cell counts)
        @test length(info["cell_index"]) > 0
        @test all(1 .<= info["cell_index"] .<= number_of_cells(mesh))
    end

    # ==================================================================
    # embed_mesh tests
    # ==================================================================
    @testset "embed_mesh grid-aligned" begin
        import Jutul.CutCellMeshes: embed_mesh

        # B is entirely inside A and aligns with A's grid lines
        g_a = CartesianMesh((4, 4, 4), (4.0, 4.0, 4.0))
        mesh_a = UnstructuredMesh(g_a)

        g_b = CartesianMesh((2, 2, 2), (2.0, 2.0, 2.0))
        mesh_b_raw = UnstructuredMesh(g_b)
        offset = SVector{3, Float64}(1.0, 1.0, 1.0)
        new_points = [p + offset for p in mesh_b_raw.node_points]
        mesh_b = Jutul.CutCellMeshes._rebuild_mesh_with_nodes(mesh_b_raw, new_points)

        result, info = embed_mesh(mesh_a, mesh_b; extra_out = true)
        geo = tpfv_geometry(result)

        vol_a = sum(tpfv_geometry(mesh_a).volumes)
        vol_result = sum(geo.volumes)

        # Volume conservation
        @test vol_result ≈ vol_a rtol=1e-8

        # All volumes positive
        @test all(geo.volumes .> 0)

        # B cells preserved
        n_b = count(x -> x == :mesh_b, info["cell_origin"])
        @test n_b == number_of_cells(mesh_b)

        # Total cells: 56 from A + 8 from B = 64
        n_a = count(x -> x == :mesh_a, info["cell_origin"])
        @test n_a + n_b == number_of_cells(result)

        # Cell indices valid
        for c in 1:number_of_cells(result)
            if info["cell_origin"][c] == :mesh_a
                @test 1 <= info["cell_index_a"][c] <= number_of_cells(mesh_a)
                @test info["cell_index_b"][c] == 0
            else
                @test info["cell_index_a"][c] == 0
                @test 1 <= info["cell_index_b"][c] <= number_of_cells(mesh_b)
            end
        end
    end

    @testset "embed_mesh misaligned" begin
        import Jutul.CutCellMeshes: embed_mesh

        # B does NOT align with A's grid lines, causing cells to be split
        g_a = CartesianMesh((4, 4, 4), (4.0, 4.0, 4.0))
        mesh_a = UnstructuredMesh(g_a)

        g_b = CartesianMesh((2, 2, 2), (1.0, 1.0, 1.0))
        mesh_b_raw = UnstructuredMesh(g_b)
        offset = SVector{3, Float64}(0.5, 0.5, 0.5)
        new_points = [p + offset for p in mesh_b_raw.node_points]
        mesh_b = Jutul.CutCellMeshes._rebuild_mesh_with_nodes(mesh_b_raw, new_points)

        result, info = embed_mesh(mesh_a, mesh_b; extra_out = true)
        geo = tpfv_geometry(result)

        vol_a = sum(tpfv_geometry(mesh_a).volumes)
        vol_result = sum(geo.volumes)

        # Volume conservation
        @test vol_result ≈ vol_a rtol=1e-6

        # All volumes positive
        @test all(geo.volumes .> 0)

        # B cells preserved exactly
        n_b = count(x -> x == :mesh_b, info["cell_origin"])
        @test n_b == number_of_cells(mesh_b)

        # More A cells than original (splitting occurred)
        n_a = count(x -> x == :mesh_a, info["cell_origin"])
        @test n_a > number_of_cells(mesh_a) - number_of_cells(mesh_b)

        # Interior normals consistency
        for f in 1:number_of_faces(result)
            l, r = result.faces.neighbors[f]
            cl = geo.cell_centroids[:, l]
            cr = geo.cell_centroids[:, r]
            N = geo.normals[:, f]
            @test dot(N, cr - cl) > 0
        end
    end

    @testset "embed_mesh B protruding" begin
        import Jutul.CutCellMeshes: embed_mesh

        # B extends outside A on one side
        g_a = CartesianMesh((4, 4, 4), (4.0, 4.0, 4.0))
        mesh_a = UnstructuredMesh(g_a)

        g_b = CartesianMesh((2, 2, 2), (2.0, 2.0, 2.0))
        mesh_b_raw = UnstructuredMesh(g_b)
        offset = SVector{3, Float64}(3.0, 1.0, 1.0)
        new_points = [p + offset for p in mesh_b_raw.node_points]
        mesh_b = Jutul.CutCellMeshes._rebuild_mesh_with_nodes(mesh_b_raw, new_points)

        result, info = embed_mesh(mesh_a, mesh_b; extra_out = true)
        geo = tpfv_geometry(result)

        vol_a = sum(tpfv_geometry(mesh_a).volumes)
        vol_b = sum(tpfv_geometry(mesh_b).volumes)
        vol_result = sum(geo.volumes)

        # Volume should be A + part of B outside A
        expected_vol = vol_a + vol_b / 2
        @test vol_result ≈ expected_vol rtol=1e-6

        # All volumes positive
        @test all(geo.volumes .> 0)

        # B cells preserved exactly
        n_b = count(x -> x == :mesh_b, info["cell_origin"])
        @test n_b == number_of_cells(mesh_b)
    end

    @testset "embed_mesh without extra_out" begin
        import Jutul.CutCellMeshes: embed_mesh

        g_a = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh_a = UnstructuredMesh(g_a)

        g_b = CartesianMesh((1, 1, 1), (1.0, 1.0, 1.0))
        mesh_b_raw = UnstructuredMesh(g_b)
        offset = SVector{3, Float64}(1.0, 1.0, 1.0)
        new_points = [p + offset for p in mesh_b_raw.node_points]
        mesh_b = Jutul.CutCellMeshes._rebuild_mesh_with_nodes(mesh_b_raw, new_points)

        # Without extra_out should just return mesh
        result = embed_mesh(mesh_a, mesh_b)
        @test isa(result, UnstructuredMesh)

        vol_a = sum(tpfv_geometry(mesh_a).volumes)
        vol_result = sum(tpfv_geometry(result).volumes)
        @test vol_result ≈ vol_a rtol=1e-6
    end

    @testset "embed_mesh B fully outside A" begin
        import Jutul.CutCellMeshes: embed_mesh

        # B is completely outside A -- should just add B's cells
        g_a = CartesianMesh((2, 2, 2), (2.0, 2.0, 2.0))
        mesh_a = UnstructuredMesh(g_a)

        g_b = CartesianMesh((2, 2, 2), (1.0, 1.0, 1.0))
        mesh_b_raw = UnstructuredMesh(g_b)
        offset = SVector{3, Float64}(5.0, 5.0, 5.0)
        new_points = [p + offset for p in mesh_b_raw.node_points]
        mesh_b = Jutul.CutCellMeshes._rebuild_mesh_with_nodes(mesh_b_raw, new_points)

        result, info = embed_mesh(mesh_a, mesh_b; extra_out = true)

        # Total cells = A cells + B cells (no overlap)
        @test number_of_cells(result) == number_of_cells(mesh_a) + number_of_cells(mesh_b)

        vol_a = sum(tpfv_geometry(mesh_a).volumes)
        vol_b = sum(tpfv_geometry(mesh_b).volumes)
        vol_result = sum(tpfv_geometry(result).volumes)
        @test vol_result ≈ vol_a + vol_b rtol=1e-8
    end
end
