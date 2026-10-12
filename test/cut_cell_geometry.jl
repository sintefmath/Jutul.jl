using Jutul, Test, LinearAlgebra, StaticArrays
import Jutul.CutCellMeshes as CCM

@testset "Shared cut-cell geometry" begin
    @testset "Polygon measures, winding and coordinate scale" begin
        for T in (Float32, Float64, BigFloat)
            lshape = SVector{2, T}.([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
            area, center = CCM.polygon_area_centroid(lshape)
            @test area ≈ T(3)
            @test center ≈ SVector{2, T}(T(5)/T(6), T(5)/T(6))
            reversed_area, reversed_center = CCM.polygon_area_centroid(reverse(lshape))
            @test reversed_area ≈ area && reversed_center ≈ center
            shifted = lshape .+ Ref(SVector{2, T}(1e7, -1e7))
            @test CCM._polygon_area_2d(shifted) ≈ T(3)
            @test CCM.polygon_area_centroid(shifted)[2] ≈ center+SVector{2, T}(1e7, -1e7) rtol = 0 atol = 2eps(T(1e7))
            parts = CCM.convex_parts(lshape, T(1e-6))
            @test sum(p -> first(CCM.polygon_area_centroid(p)), parts) ≈ T(3)
            # The vertex mean lies in this U-shaped polygon's notch. An
            # unsigned fan overestimates its area and changes the centroid.
            ushape = SVector{3, T}[(3, 0, 0), (3, 3, 0), (3, 3, 3), (3, 2, 3),
                (3, 2, 1), (3, 1, 1), (3, 1, 3), (3, 0, 3)]
            @test CCM.polygon_area(ushape) ≈ T(7)
            @test CCM.polygon_centroid(ushape) ≈ SVector{3, T}(3, T(3)/T(2), T(19)/T(14))
            @test CCM.polygon_normal(ushape) ≈ SVector{3, T}(1, 0, 0)
        end
        empty = SVector{2, Float64}[]
        @test CCM.polygon_area_centroid(empty) == (0, SVector(0.0, 0.0))
        @test isempty(CCM.convex_parts(empty, 1e-9))
        crossing = SVector{2, Float64}.([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0),
            (1, 1), (3, 1), (3, 3), (1, 3), (1, 1), (0, 0)])
        @test sum(p -> first(CCM.polygon_area_centroid(p)),
            CCM.convex_parts(crossing, 1e-9; resolve_winding = true)) ≈ 7
        warped = SVector{3, Float64}[(0, 0, 0), (1, 0, 0), (1, 1, 0.3), (0, 1, 0)]
        center, area = Jutul.face_centroid_and_measure(collect(1:4), warped)
        @test CCM.polygon_area(warped) ≈ area
        @test CCM.polygon_centroid(warped) ≈ center
        small_warped = SVector{3, Float32}.(warped).*Float32(1e-8)
        small_center, small_area = Jutul.face_centroid_and_measure(collect(1:4), small_warped)
        @test CCM.polygon_area(small_warped) ≈ small_area
        @test CCM.polygon_centroid(small_warped) ≈ small_center
    end

    @testset "Clipping and legacy projection interfaces" begin
        for T in (Float32, Float64, BigFloat), scale in T.((1e-8, 1.0, 1e8))
            a = scale.*SVector{2, T}.([(0, 0), (2, 0), (2, 2), (0, 2)])
            b = scale.*SVector{2, T}.([(1, -1), (3, -1), (3, 1), (1, 1)])
            overlap = CCM._polygon_intersection_2d(reverse(a), reverse(b))
            area, center = CCM.polygon_area_centroid(overlap)
            @test area/scale^2 ≈ one(T) rtol = 32eps(T)
            @test center/scale ≈ SVector{2, T}(1.5, 0.5) rtol = 32eps(T)
            inside, outside = CCM._split_polygon_by_line_2d(a,
                SVector{2, T}(scale, 0), SVector{2, T}(1, 0))
            @test CCM.polygon_area_centroid(inside)[1]/scale^2 ≈ T(2)
            @test CCM.polygon_area_centroid(outside)[1]/scale^2 ≈ T(2)
            tiny_normal = SVector{3, T}(T(1e-30), 0, 0)
            @test CCM.PlaneCut(zero(SVector{3, T}), tiny_normal).normal == SVector{3, T}(1, 0, 0)
        end
        plane = CCM.PlaneCut([0, 0, 0], [0, 0, 1])
        @test CCM.project_to_plane(SVector(1.0, 2.0, 3.0), plane) == SVector(2.0, -1.0)
        @test CCM._build_tangent_basis(plane.normal) == (SVector(0.0, 1.0, 0.0), SVector(-1.0, 0.0, 0.0))
        @test CCM.plane_basis(plane.normal; convention = :axis) == (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0))
        @test_throws ArgumentError CCM.PlaneCut(zeros(3), zeros(3))
        integer_a = SVector{2, Int}[(0, 0), (2, 0), (2, 2), (0, 2)]
        integer_b = SVector{2, Int}[(1, -1), (3, -1), (3, 1), (1, 1)]
        @test CCM._polygon_area_2d(CCM._polygon_intersection_2d(integer_a, integer_b)) == 1
    end

    @testset "Frames, sections, distance integration and extrusion" begin
        points = SVector{3, Float64}[(0, 0, 0.5), (2, 0, 0.5), (2, 1, 0.5),
            (1, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)]
        frame = CCM.PlanarFrame(points)
        area, center = CCM.polygon_parts_area_centroid(frame.parts, frame)
        @test area ≈ 3
        @test center ≈ SVector(5/6, 5/6, 0.5)
        @test CCM.unproject(frame, CCM.project(frame, points[3])) ≈ points[3]
        @test_throws ArgumentError CCM.PlanarFrame([SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0),
            SVector(1.0, 1.0, 0.2), SVector(0.0, 1.0, 0.0)])
        vertices, faces = CCM.polygon_prism(frame.parts, frame, 0.1, 1e-9)
        # A closed prism must use every edge in both orientations exactly once.
        edges = Dict{Tuple{Int, Int}, Int}()
        for face in faces, i in eachindex(face)
            edge = (face[i], face[mod1(i+1, length(face))])
            edges[edge] = get(edges, edge, 0)+1
        end
        @test all(e -> edges[e] == get(edges, reverse(e), 0), keys(edges))
        @test extrema(p -> p[3], vertices) == (0.45, 0.55)
        mesh = UnstructuredMesh(CartesianMesh((1, 1, 1)))
        origin = SVector(0.5, 0.5, 0.5)
        normal = SVector(1.0, 0.0, 0.0)
        u, v = CCM.plane_basis(normal; convention = :axis)
        loops, segments = CCM.cell_section(mesh, 1, origin, normal, u, v, 1e-9)
        @test sum(p -> CCM.polygon_area_centroid(p)[1], loops) ≈ 1
        @test CCM.cell_average_distance(mesh, 1, origin, normal) ≈ 0.25
        @test CCM.segment_overlap(origin, origin, origin, origin, 1e-9) === nothing
    end

    @testset "Spatial tree queries" begin
        for T in (Float32, Float64, BigFloat)
            mesh = UnstructuredMesh(CartesianMesh((3, 2, 2), (3.0, 2.0, 2.0)))
            index = CCM.CellSpatialIndex(mesh, T)
            low, high = SVector{3, T}(0, 0, 0), SVector{3, T}(3, 2, 2)
            candidates = CCM.query_cells(index, low, high, T(1e-9);
                origin = SVector{3, T}(1.5, 1, 1), normal = SVector{3, T}(1, 0, 0))
            @test candidates == [2, 5, 8, 11]
            empty = CCM.CellSpatialIndex(SVector{3, T}[], SVector{3, T}[])
            @test isempty(CCM.query_cells(empty, low, high, T(1e-9)))
        end
        points = [SVector(0.0, 0.0, 0.0)]
        @test_throws ArgumentError CCM.CellSpatialIndex(points, points; leaf_size = 0)
        @test_throws ArgumentError CCM.CellSpatialIndex(points, SVector{3, Float64}[])
    end
end
