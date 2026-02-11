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
            constant = 0.0, slope = 0.0, side = :positive,
            tol = 1e-6, face_tol = 1.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-8
    end

    @testset "large constant displacement" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.5, slope = 0.0, side = :positive,
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
            constant = 0.5, slope = 0.0, side = :negative,
            tol = 1e-6, face_tol = 2.0, min_cut_fraction = 0.01
        )

        @test number_of_cells(result) > number_of_cells(mesh)
        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end

    @testset "slope preserves cell volumes" begin
        # The slope displacement shifts each node perpendicular to the plane by
        # dz = (x - x0) * slope.  This is equivalent to a shear transform which
        # preserves volumes of the shifted half.
        g = CartesianMesh((4, 4, 4), (4.0, 4.0, 4.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([2.0, 2.0, 2.5], [0.0, 0.0, 1.0])
        # First cut and extract to get the half that will be shifted
        cut, cut_info = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)
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
        vol_pos_before = sum(tpfv_geometry(mesh_pos).volumes)

        # Apply the same shear transform that cut_and_displace_mesh uses
        n = plane.normal
        ref = abs(n[1]) < 0.9 ? SVector{3, Float64}(1, 0, 0) : SVector{3, Float64}(0, 1, 0)
        tangent = normalize(cross(n, ref))
        x0 = dot(plane.point, tangent)

        shifted_nodes = copy(mesh_pos.node_points)
        for i in eachindex(shifted_nodes)
            pt = shifted_nodes[i]
            x = dot(pt, tangent)
            dz = (x - x0) * 0.5
            shifted_nodes[i] = pt + dz * n
        end
        import Jutul.CutCellMeshes: _rebuild_mesh_with_nodes
        shifted_mesh = _rebuild_mesh_with_nodes(mesh_pos, shifted_nodes)
        vol_pos_after = sum(tpfv_geometry(shifted_mesh).volumes)

        # Volume of the shifted half must be exactly preserved (shear transform)
        @test vol_pos_after ≈ vol_pos_before rtol = 1e-12

        # Also run the full pipeline
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.0, slope = 0.5, side = :positive,
            tol = 1e-6, face_tol = 5.0, min_cut_fraction = 0.01
        )
        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end

    @testset "large slope does not shrink mesh" begin
        # With a large slope, a tangent-direction shift would shrink/stretch the
        # mesh. The normal-direction shift (shear) should preserve volumes.
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])

        # Extract the shifted half and verify volume is preserved
        cut, _ = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)
        pos_cells = Int[]
        for c in 1:number_of_cells(cut)
            cl = classify_cell(cut, c, plane; tol = 1e-6)
            if cl == :positive
                push!(pos_cells, c)
            end
        end
        mesh_pos = extract_submesh(cut, pos_cells)
        vol_pos_before = sum(tpfv_geometry(mesh_pos).volumes)

        # Apply shear with large slope
        n = plane.normal
        ref = abs(n[1]) < 0.9 ? SVector{3, Float64}(1, 0, 0) : SVector{3, Float64}(0, 1, 0)
        tangent = normalize(cross(n, ref))
        x0 = dot(plane.point, tangent)

        shifted_nodes = copy(mesh_pos.node_points)
        for i in eachindex(shifted_nodes)
            pt = shifted_nodes[i]
            x = dot(pt, tangent)
            dz = (x - x0) * 1.0
            shifted_nodes[i] = pt + dz * n
        end
        shifted_mesh = _rebuild_mesh_with_nodes(mesh_pos, shifted_nodes)
        vol_pos_after = sum(tpfv_geometry(shifted_mesh).volumes)

        # Volume of the shifted half must be preserved
        @test vol_pos_after ≈ vol_pos_before rtol = 1e-12

        # Full pipeline should produce valid geometry
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.0, slope = 1.0, side = :positive,
            tol = 1e-6, face_tol = 4.0, min_cut_fraction = 0.01
        )
        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end

    @testset "combined constant and large slope" begin
        g = CartesianMesh((4, 4, 4), (4.0, 4.0, 4.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([2.0, 2.0, 2.5], [0.0, 0.0, 1.0])

        # Extract shifted half for volume check
        cut, _ = cut_mesh(mesh, plane; min_cut_fraction = 0.01, extra_out = true)
        pos_cells = Int[]
        for c in 1:number_of_cells(cut)
            cl = classify_cell(cut, c, plane; tol = 1e-6)
            if cl == :positive
                push!(pos_cells, c)
            end
        end
        mesh_pos = extract_submesh(cut, pos_cells)
        vol_pos_before = sum(tpfv_geometry(mesh_pos).volumes)

        n = plane.normal
        ref = abs(n[1]) < 0.9 ? SVector{3, Float64}(1, 0, 0) : SVector{3, Float64}(0, 1, 0)
        tangent = normalize(cross(n, ref))
        x0 = dot(plane.point, tangent)

        # constant goes as tangent, slope goes as normal
        shifted_nodes = copy(mesh_pos.node_points)
        for i in eachindex(shifted_nodes)
            pt = shifted_nodes[i]
            x = dot(pt, tangent)
            shifted_nodes[i] = pt + 0.3 * tangent + (x - x0) * 0.8 * n
        end
        shifted_mesh = _rebuild_mesh_with_nodes(mesh_pos, shifted_nodes)
        vol_pos_after = sum(tpfv_geometry(shifted_mesh).volumes)

        # Volume of the shifted half must be preserved (tangent translation +
        # normal shear both preserve volume)
        @test vol_pos_after ≈ vol_pos_before rtol = 1e-12

        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.3, slope = 0.8, side = :positive,
            tol = 1e-6, face_tol = 5.0, min_cut_fraction = 0.01
        )
        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end

    @testset "oblique plane constant displacement preserves volume" begin
        # This is the case from the problem statement: an oblique cut plane
        # with a constant displacement should slide along the plane (not away),
        # keeping the interface in contact and preserving volume.
        g = CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([0.5, 0.5, 0.5], [1.0, 0.2, 0.1])
        result = cut_and_displace_mesh(mesh, plane;
            constant = 0.4, side = :positive,
            slope = 0.0,
            tol = 1e-6, face_tol = 1.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        # Volume must be conserved: tangential shift keeps the interface in
        # contact, so no gap is created.
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-6
    end

    @testset "extra_out maps back to original mesh" begin
        g = CartesianMesh((3, 3, 3), (3.0, 3.0, 3.0))
        mesh = UnstructuredMesh(g)
        nc_orig = number_of_cells(mesh)

        plane = PlaneCut([1.5, 1.5, 1.5], [0.0, 0.0, 1.0])
        result, info = cut_and_displace_mesh(mesh, plane;
            constant = 0.0, slope = 0.0, side = :positive,
            tol = 1e-6, face_tol = 2.0, min_cut_fraction = 0.01,
            extra_out = true
        )

        nc = number_of_cells(result)
        nf = number_of_faces(result)
        nb = number_of_boundary_faces(result)

        # cell_index maps every new cell back to an original cell
        @test length(info["cell_index"]) == nc
        @test all(1 .<= info["cell_index"] .<= nc_orig)

        # cell_side tells us which side each cell is on
        @test length(info["cell_side"]) == nc
        @test all(s -> s == :positive || s == :negative, info["cell_side"])

        # Both sides must be present
        @test any(s -> s == :positive, info["cell_side"])
        @test any(s -> s == :negative, info["cell_side"])

        # face/boundary tracking arrays have correct lengths
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
            constant = 0.5, slope = 0.5, side = :positive,
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
            constant = 0.0, slope = 0.0, side = :positive,
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
            constant = 0.0, slope = 0.0, side = :positive,
            tol = 1e-8, face_tol = 1.5, min_cut_fraction = 0.01,
            area_tol = 1e-12
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end
end
