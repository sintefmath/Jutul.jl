using Jutul
using Test
using LinearAlgebra
using StaticArrays

import Jutul.CutCellMeshes: PlaneCut, cut_mesh, glue_mesh, mesh_fault_slip, classify_cell

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

@testset "Mesh Fault Slip" begin
    @testset "mesh_fault_slip zero displacement" begin
        g = CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
        result = mesh_fault_slip(mesh, plane;
            constant = 0.0, slope = 0.0, side = :positive,
            tol = 1e-6, face_tol = 1.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-8
    end

    @testset "mesh_fault_slip constant displacement" begin
        g = CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
        result = mesh_fault_slip(mesh, plane;
            constant = 0.1, slope = 0.0, side = :positive,
            tol = 1e-6, face_tol = 1.0, min_cut_fraction = 0.01
        )

        @test number_of_cells(result) > number_of_cells(mesh)
        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end

    @testset "mesh_fault_slip negative side" begin
        g = CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
        result = mesh_fault_slip(mesh, plane;
            constant = 0.1, slope = 0.0, side = :negative,
            tol = 1e-6, face_tol = 1.0, min_cut_fraction = 0.01
        )

        @test number_of_cells(result) > number_of_cells(mesh)
        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end

    @testset "mesh_fault_slip linear displacement" begin
        g = CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
        result = mesh_fault_slip(mesh, plane;
            constant = 0.05, slope = 0.1, side = :positive,
            tol = 1e-6, face_tol = 1.0, min_cut_fraction = 0.01
        )

        @test number_of_cells(result) > number_of_cells(mesh)
        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end

    @testset "mesh_fault_slip extra_out" begin
        g = CartesianMesh((3, 3, 3), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
        result, info = mesh_fault_slip(mesh, plane;
            constant = 0.0, slope = 0.0, side = :positive,
            tol = 1e-6, face_tol = 1.0, min_cut_fraction = 0.01,
            extra_out = true
        )

        nc = number_of_cells(result)
        nf = number_of_faces(result)
        nb = number_of_boundary_faces(result)

        @test length(info["cell_index_a"]) == nc
        @test length(info["cell_index_b"]) == nc
        @test length(info["face_index_a"]) == nf
        @test length(info["face_index_b"]) == nf
        @test length(info["boundary_face_index_a"]) == nb
        @test length(info["boundary_face_index_b"]) == nb
    end

    @testset "mesh_fault_slip x-normal plane" begin
        g = CartesianMesh((4, 3, 3), (2.0, 1.5, 1.5))
        mesh = UnstructuredMesh(g)
        vol_orig = sum(tpfv_geometry(mesh).volumes)

        plane = PlaneCut([0.6, 0.75, 0.75], [1.0, 0.0, 0.0])
        result = mesh_fault_slip(mesh, plane;
            constant = 0.0, slope = 0.0, side = :positive,
            tol = 1e-6, face_tol = 1.0, min_cut_fraction = 0.01
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
        @test sum(geo.volumes) ≈ vol_orig rtol = 1e-8
    end

    @testset "mesh_fault_slip tolerances" begin
        g = CartesianMesh((2, 2, 2), (1.0, 1.0, 1.0))
        mesh = UnstructuredMesh(g)

        plane = PlaneCut([0.5, 0.5, 0.5], [0.0, 0.0, 1.0])

        # Use different tolerances and verify it still works
        result = mesh_fault_slip(mesh, plane;
            constant = 0.0, slope = 0.0, side = :positive,
            tol = 1e-8, face_tol = 0.5, min_cut_fraction = 0.01,
            area_tol = 1e-12
        )

        geo = tpfv_geometry(result)
        @test all(geo.volumes .> 0)
    end
end
