using Jutul
using Test
using LinearAlgebra
using StaticArrays

import Jutul.CutCellMeshes: PlaneCut, PolygonalSurface, cut_mesh

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
end
