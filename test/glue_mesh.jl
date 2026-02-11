using Jutul
using Test
using LinearAlgebra
using StaticArrays

import Jutul.CutCellMeshes: PlaneCut, cut_mesh, glue_mesh, cut_and_displace_mesh, classify_cell

@testset "Mesh Gluing" begin
    @testset "glue_mesh basic - cut and rejoin" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut, _ = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)

        pos_cells = Int[]
        neg_cells = Int[]
        for c in 1:number_of_cells(cut)
            cl = classify_cell(cut, c, plane; tol = 1e-6)
            if cl == :positive
                push!(pos_cells, c)
            else
                push!(neg_cells, c)
            end
        end

        mesh_pos = extract_submesh(cut, pos_cells)
        mesh_neg = extract_submesh(cut, neg_cells)

        glued = glue_mesh(mesh_pos, mesh_neg; tol = 1e-6, face_tol = 1.0)

        @test number_of_cells(glued) == number_of_cells(mesh_pos) + number_of_cells(mesh_neg)

        geo = tpfv_geometry(glued)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-8
    end

    @testset "glue_mesh normal consistency" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut, _ = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)

        pos_cells = Int[]
        neg_cells = Int[]
        for c in 1:number_of_cells(cut)
            cl = classify_cell(cut, c, plane; tol = 1e-6)
            if cl == :positive
                push!(pos_cells, c)
            else
                push!(neg_cells, c)
            end
        end

        mesh_pos = extract_submesh(cut, pos_cells)
        mesh_neg = extract_submesh(cut, neg_cells)

        glued = glue_mesh(mesh_pos, mesh_neg; tol = 1e-6, face_tol = 1.0)
        geo = tpfv_geometry(glued)

        for f in 1:number_of_faces(glued)
            l, r = glued.faces.neighbors[f]
            cl = geo.cell_centroids[:, l]
            cr = geo.cell_centroids[:, r]
            N = geo.normals[:, f]
            @test dot(N, cr - cl) > 0
        end

        for f in 1:number_of_boundary_faces(glued)
            c = glued.boundary_faces.neighbors[f]
            cc = geo.cell_centroids[:, c]
            fc = geo.boundary_centroids[:, f]
            N = geo.boundary_normals[:, f]
            @test dot(N, fc - cc) > 0
        end
    end

    @testset "glue_mesh extra_out" begin
        g = CartesianMesh((3, 3, 3))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])
        cut, _ = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)

        pos_cells = Int[]
        neg_cells = Int[]
        for c in 1:number_of_cells(cut)
            cl = classify_cell(cut, c, plane; tol = 1e-6)
            if cl == :positive
                push!(pos_cells, c)
            else
                push!(neg_cells, c)
            end
        end

        mesh_pos = extract_submesh(cut, pos_cells)
        mesh_neg = extract_submesh(cut, neg_cells)
        nc_a = number_of_cells(mesh_pos)
        nc_b = number_of_cells(mesh_neg)

        glued, info = glue_mesh(mesh_pos, mesh_neg;
            tol = 1e-6, face_tol = 1.0, extra_out = true)

        nc = number_of_cells(glued)
        nf = number_of_faces(glued)
        nb = number_of_boundary_faces(glued)

        @test length(info["cell_index_a"]) == nc
        @test length(info["cell_index_b"]) == nc
        @test length(info["face_index_a"]) == nf
        @test length(info["face_index_b"]) == nf
        @test length(info["boundary_face_index_a"]) == nb
        @test length(info["boundary_face_index_b"]) == nb

        # Cell mappings: first nc_a map to mesh_a, rest to mesh_b
        @test all(info["cell_index_a"][1:nc_a] .== 1:nc_a)
        @test all(info["cell_index_a"][nc_a+1:end] .== 0)
        @test all(info["cell_index_b"][1:nc_a] .== 0)
        @test all(info["cell_index_b"][nc_a+1:end] .== 1:nc_b)

        # New faces should have zero origin indices
        @test length(info["new_faces"]) == 9
        for f in info["new_faces"]
            @test info["face_index_a"][f] == 0
            @test info["face_index_b"][f] == 0
        end
    end

    @testset "glue_mesh different cut planes" begin
        for normal in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
            g = CartesianMesh((2, 2, 2), (2.0, 2.0, 2.0))
            mesh = UnstructuredMesh(g)
            vol_orig = sum(tpfv_geometry(mesh).volumes)

            plane = PlaneCut([1.0, 1.0, 0.5], normal)
            cut, _ = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)

            pos_cells = Int[]
            neg_cells = Int[]
            for c in 1:number_of_cells(cut)
                cl = classify_cell(cut, c, plane; tol = 1e-6)
                if cl == :positive
                    push!(pos_cells, c)
                else
                    push!(neg_cells, c)
                end
            end

            mesh_pos = extract_submesh(cut, pos_cells)
            mesh_neg = extract_submesh(cut, neg_cells)

            glued = glue_mesh(mesh_pos, mesh_neg; tol = 1e-6, face_tol = 2.5)
            geo = tpfv_geometry(glued)
            @test all(geo.volumes .> 0)
            @test sum(geo.volumes) ≈ vol_orig rtol = 1e-8
        end
    end
end

@testset "cut_and_displace_mesh" begin
    @testset "zero displacement" begin
        g = CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.0, side = :positive,
            tol = 1e-6, face_tol = 1.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-8
    end

    @testset "large constant displacement (t1 shift)" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.5, side = :positive,
            tol = 1e-6, face_tol = 2.0, min_cut_fraction = 0.01
        )

        @test number_of_cells(result) > number_of_cells(mesh)
        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end

    @testset "negative side displacement" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.5, side = :negative,
            tol = 1e-6, face_tol = 2.0, min_cut_fraction = 0.01
        )

        @test number_of_cells(result) > number_of_cells(mesh)
        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end

    @testset "shift_lr displacement (t2 shift)" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result = cut_and_displace_mesh(mesh, plane;
            shift_lr = 0.5, side = :positive,
            tol = 1e-6, face_tol = 2.0, min_cut_fraction = 0.01
        )

        @test number_of_cells(result) > number_of_cells(mesh)
        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end

    @testset "constant preserves volume (interface stays in contact)" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.5, side = :positive,
            tol = 1e-6, face_tol = 4.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-6
    end

    @testset "shift_lr preserves volume (interface stays in contact)" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result = cut_and_displace_mesh(mesh, plane;
            shift_lr = 0.5, side = :positive,
            tol = 1e-6, face_tol = 4.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-6
    end

    @testset "angle preserves volume (interface stays in contact)" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        # Small angle (15 degrees)
        result = cut_and_displace_mesh(mesh, plane;
            angle = π / 12, side = :positive,
            tol = 1e-6, face_tol = 4.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-6
    end

    @testset "angle rotation is in-plane (no normal displacement)" begin
        # Verify that rotation around the plane normal only moves nodes
        # within the plane — the normal component stays unchanged.
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        cut, _ = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)
        pos_cells = Int[]
        for c in 1:number_of_cells(cut)
            cl = classify_cell(cut, c, plane; tol = 1e-6)
            if cl == :positive
                push!(pos_cells, c)
            end
        end
        mesh_pos = extract_submesh(cut, pos_cells)

        n = plane.normal
        ref = abs(n[1]) < 0.9 ? SVector{3, Float64}(1, 0, 0) : SVector{3, Float64}(0, 1, 0)
        t1 = normalize(cross(n, ref))
        t2 = cross(n, t1)

        θ = π / 6  # 30 degrees
        cosθ = cos(θ)
        sinθ = sin(θ)

        for i in eachindex(mesh_pos.node_points)
            pt = mesh_pos.node_points[i]
            dp = pt - plane.point
            d_before = dot(dp, n)
            x1 = dot(dp, t1)
            x2 = dot(dp, t2)
            x1_new = x1 * cosθ - x2 * sinθ
            x2_new = x1 * sinθ + x2 * cosθ
            pt_new = plane.point + x1_new * t1 + x2_new * t2 + d_before * n
            d_after = dot(pt_new - plane.point, n)
            # Normal distance must be unchanged
            @test d_after ≈ d_before atol = 1e-12
        end
    end

    @testset "angle preserves cell volumes of shifted half" begin
        # Rotation is volume-preserving: the shifted half should have
        # the same total volume before and after.
        g = CartesianMesh((4, 4, 4), (4.0, 4.0, 4.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([2.0, 2.0, 2.0], [0.0, 0.0, 1.0])
        cut, _ = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)
        pos_cells = Int[]
        for c in 1:number_of_cells(cut)
            cl = classify_cell(cut, c, plane; tol = 1e-6)
            if cl == :positive
                push!(pos_cells, c)
            end
        end
        mesh_pos = extract_submesh(cut, pos_cells)
        vol_before = sum(tpfv_geometry(mesh_pos).volumes)

        n = plane.normal
        ref = abs(n[1]) < 0.9 ? SVector{3, Float64}(1, 0, 0) : SVector{3, Float64}(0, 1, 0)
        t1 = normalize(cross(n, ref))
        t2 = cross(n, t1)

        θ = π / 4  # 45 degrees
        cosθ = cos(θ)
        sinθ = sin(θ)

        import Jutul.CutCellMeshes: _rebuild_mesh_with_nodes
        shifted_nodes = copy(mesh_pos.node_points)
        for i in eachindex(shifted_nodes)
            pt = shifted_nodes[i]
            dp = pt - plane.point
            x1 = dot(dp, t1)
            x2 = dot(dp, t2)
            d  = dot(dp, n)
            x1_new = x1 * cosθ - x2 * sinθ
            x2_new = x1 * sinθ + x2 * cosθ
            shifted_nodes[i] = plane.point + x1_new * t1 + x2_new * t2 + d * n
        end
        shifted_mesh = _rebuild_mesh_with_nodes(mesh_pos, shifted_nodes)
        vol_after = sum(tpfv_geometry(shifted_mesh).volumes)

        @test vol_after ≈ vol_before rtol = 1e-12
    end

    @testset "combined constant and shift_lr" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.3, shift_lr = 0.4, side = :positive,
            tol = 1e-6, face_tol = 4.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-6
    end

    @testset "combined constant, shift_lr and angle" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.3, shift_lr = 0.2, angle = π / 12,
            side = :positive,
            tol = 1e-6, face_tol = 4.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-6
    end

    @testset "oblique plane constant displacement preserves volume" begin
        g = CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([0.5, 0.5, 0.5], [1.0, 0.2, 0.1])
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.4, side = :positive,
            tol = 1e-6, face_tol = 1.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-6
    end

    @testset "extra_out maps back to original mesh" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result, info = cut_and_displace_mesh(mesh, plane;
            constant = 0.0, side = :positive,
            tol = 1e-6, face_tol = 2.0, min_cut_fraction = 0.01,
            extra_out = true
        )

        nc = number_of_cells(result)
        nf = number_of_faces(result)
        nb = number_of_boundary_faces(result)

        @test length(info["cell_index"]) == nc
        @test all(1 .<= info["cell_index"] .<= nc_orig)

        @test length(info["cell_side"]) == nc
        @test all(s -> s == :positive || s == :negative, info["cell_side"])

        @test any(s -> s == :positive, info["cell_side"])
        @test any(s -> s == :negative, info["cell_side"])

        @test length(info["face_index_a"]) == nf
        @test length(info["face_index_b"]) == nf
        @test length(info["boundary_face_index_a"]) == nb
        @test length(info["boundary_face_index_b"]) == nb
    end

    @testset "extra_out with displacement" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result, info = cut_and_displace_mesh(mesh, plane;
            constant = 0.5, shift_lr = 0.3, side = :positive,
            tol = 1e-6, face_tol = 4.0, min_cut_fraction = 0.01,
            extra_out = true
        )

        nc = number_of_cells(result)
        @test length(info["cell_index"]) == nc
        @test all(1 .<= info["cell_index"] .<= nc_orig)
        @test length(info["cell_side"]) == nc
        @test any(s -> s == :positive, info["cell_side"])
        @test any(s -> s == :negative, info["cell_side"])
    end

    @testset "x-normal plane" begin
        g = CartesianMesh((4, 3, 3), (4.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([1.2, 1.5, 1.5], [1.0, 0.0, 0.0])
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.0, side = :positive,
            tol = 1e-6, face_tol = 2.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-8
    end

    @testset "tolerances" begin
        g = CartesianMesh((2, 2, 2), (2.0, 2.0, 2.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([1.0, 1.0, 1.0], [0.0, 0.0, 1.0])

        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.0, side = :positive,
            tol = 1e-8, face_tol = 1.5, min_cut_fraction = 0.01,
            area_tol = 1e-12
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end
end
