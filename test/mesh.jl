using Jutul
using Test
using LinearAlgebra

@testset "CartesianMesh indices" begin
    for dims in [(3, 5, 7), (5, 7), (8,)]
        # dims = (3, 5, 7)
        n = prod(dims)
        g = CartesianMesh(dims)
        d = length(dims)
        @testset "Basic consistency" begin
            @test number_of_cells(g) == n
            @test grid_dims_ijk(g)[1:d] == dims
        end
        nx = dims[1]
        ny = nz = 1
        if d > 1
            ny = dims[2]
            if d > 2
                nz = dims[3]
            end
        end
        @testset "$d-D IJK -> linear indexing" begin
            lix = 1
            for k = 1:nz
                for j = 1:ny
                    for i = 1:nx
                        @test cell_index(g, (i, j, k)) == lix
                        lix += 1
                    end
                end
            end
        end
        @testset "$d-D linear -> IJK indexing" begin
            lix = 1
            for k = 1:nz
                for j = 1:ny
                    for i = 1:nx
                        @test cell_ijk(g, lix) == (i, j, k)
                        lix += 1
                    end
                end
            end
        end
    end
end

using MAT
@testset "UnstructuredMesh" begin
    exported = Jutul.get_mat_testgrid("pico")
    G_raw = exported["G"]
    g = MRSTWrapMesh(G_raw)
    G = UnstructuredMesh(g)
    @testset "basics" begin
        function test_faces(G, g)
            for i = 1:number_of_faces(G)
                f_ix = G.face_map[i]
                if f_ix > 0
                    e = Faces()
                else
                    e = BoundaryFaces()
                end
                c, a = Jutul.compute_centroid_and_measure(G, e, abs(f_ix))
                c_mrst = G_raw["faces"]["centroids"][i, :]
                a_mrst = G_raw["faces"]["areas"][i]
                @test c_mrst ≈ c
                @test a_mrst ≈ a
            end
            N_raw = g.data.faces.neighbors'
            num_bfaces = 0
            num_faces = 0
            for i in axes(N_raw, 2)
                if N_raw[1, i] == 0 || N_raw[2, i] == 0
                    num_bfaces += 1
                else
                    num_faces += 1
                end
            end
            @test num_faces == number_of_faces(G)
            @test num_bfaces == number_of_boundary_faces(G)
        end
        test_faces(G, g)

        for i = 1:number_of_cells(G)
            c, v = Jutul.compute_centroid_and_measure(G, Cells(), i)
            c_mrst = G_raw["cells"]["centroids"][i, :]
            v_mrst = G_raw["cells"]["volumes"][i]
            @test c_mrst ≈ c
            @test v_mrst ≈ v
        end
    end
    geo = tpfv_geometry(G)
    geo_mrst = tpfv_geometry(g)
    @testset "geometry" begin
        @test geo_mrst.neighbors == geo.neighbors
        @test geo_mrst.areas ≈ geo.areas
        @test geo_mrst.normals ≈ geo.normals
        @test geo_mrst.face_centroids ≈ geo.face_centroids
        @test geo_mrst.volumes ≈ geo.volumes
        @test geo_mrst.cell_centroids ≈ geo.cell_centroids
    end

    @testset "cartesian to unstructured" begin
        meshes_1d = [
            CartesianMesh((3,)),
            CartesianMesh((3,), ([1.0, 3.0, 4.0], )),
        ]
        meshes_2d = [
            CartesianMesh((3, 2)),
            CartesianMesh((3, 2), ([1.0, 3.0, 4.0], [1.0, 2.0])),
        ]
        meshes_3d = [
            CartesianMesh((3, 2, 2), ((1.0, 3.0, 4.0), (1.0, 2.0), 1.0)),
            CartesianMesh((3, 2, 2)),
            CartesianMesh((4, 1, 1)),
            CartesianMesh((9, 7, 5), origin = [0.2, 0.6, 10.1]),
            CartesianMesh((3, 2, 2), (10.0, 3.0, 5.0)),
            CartesianMesh((3, 2, 2), ([10.0, 5.0, π], 3.0, 5.0)),
            CartesianMesh((100, 3, 7))
        ]
        for mdim in 1:3
            @testset "$(mdim)D conversion" begin
                if mdim == 1
                    test_meshes = meshes_1d
                    subs = 1:1
                else
                    if mdim == 2
                        test_meshes = meshes_2d
                    else
                        test_meshes = meshes_3d
                    end
                    subs = 1:mdim
                end
                for g in test_meshes
                    G = UnstructuredMesh(g, warn_1d = false)
                    geo1 = tpfv_geometry(g)
                    geo2 = tpfv_geometry(G)

                    @testset "cells" begin
                        @test geo1.volumes ≈ geo2.volumes
                        @test geo1.cell_centroids ≈ geo2.cell_centroids[1:mdim, :]
                    end
                    @testset "faces" begin
                        @test geo1.neighbors == geo2.neighbors
                        @test geo1.normals == geo2.normals[1:mdim, :]
                        @test geo1.areas ≈ geo2.areas
                        @test geo1.face_centroids ≈ geo2.face_centroids[1:mdim, :]
                    end
                    if mdim > 1
                        @testset "boundary" begin
                            # Note: 1D grids are currently converted to 2D, so
                            # the boundary will not be the same after
                            # conversion.
                            @test geo1.boundary_normals == geo2.boundary_normals[1:mdim, :]
                            @test geo1.boundary_neighbors == geo2.boundary_neighbors
                            @test geo1.boundary_areas ≈ geo2.boundary_areas
                            @test geo1.boundary_centroids ≈ geo2.boundary_centroids[1:mdim, :]
                        end
                    end
                    @testset "half-faces" begin
                        @test geo1.half_face_faces == geo2.half_face_faces
                        @test geo1.half_face_cells == geo2.half_face_cells
                    end
                    @testset "triangulate_mesh" begin
                        try
                            triangulate_mesh(G)
                        catch
                            @test false
                        finally
                            @test true
                        end
                    end
                end
            end
        end
        # 1D support missing
        @test_warn "Conversion from CartesianMesh to UnstructuredMesh is only fully supported for 2D/3D grids. Converting 1D grid to 2D." UnstructuredMesh(CartesianMesh((3,)))
        @test_nowarn UnstructuredMesh(CartesianMesh((3,)), warn_1d = false)
    end
    @testset "extract_submesh + cart convert" begin
        g = CartesianMesh((2, 2, 2))
        G = UnstructuredMesh(g)
        G_sub = extract_submesh(g, 1:3)
        @test number_of_cells(G_sub) == 3
        @test number_of_faces(G_sub) == 2
    end
end

@testset "CoarseMesh" begin
    G = CartesianMesh((4, 1, 1))
    uG = UnstructuredMesh(G)
    p = [2, 2, 1, 1]

    CG = CoarseMesh(G, p)
    geo = tpfv_geometry(G)
    geo_c = tpfv_geometry(CG)

    @test geo.volumes[1] ≈ geo_c.volumes[1]/2
    # Make a trivial coarse grid and test
    CG2 = CoarseMesh(G, [1, 2, 3, 4])
    geo_c2 = tpfv_geometry(CG2)

    for f in [:neighbors, :boundary_neighbors, :half_face_cells, :half_face_faces]
        @test getfield(geo_c2, f) == getfield(geo, f)
    end
    for f in [:normals, :boundary_centroids, :boundary_normals, :boundary_areas, :cell_centroids, :face_centroids, :volumes, :areas]
        @test getfield(geo_c2, f) ≈ getfield(geo, f)
    end
end

import Jutul: cells_inside_bounding_box
@testset "Trajectories" begin
    G = CartesianMesh((4, 4, 5), (100.0, 100.0, 100.0))
    trajectory = [
        50.0 25.0 1;
        55 35.0 25;
        65.0 40.0 50.0;
        70.0 70.0 90.0
    ]
    cells_3d = Jutul.find_enclosing_cells(G, trajectory)
    @test cells_3d == [7, 23, 39, 55, 59, 75]

    G = CartesianMesh((5, 5), (1.0, 2.0))
    trajectory = [
        0.1 0.1;
        0.25 0.4;
        0.3 1.2
    ]
    cells_2d = Jutul.find_enclosing_cells(G, trajectory)
    @test cells_2d == [1, 2, 7, 12]
    @testset "cells_inside_bounding_box" begin
        tm = convert(UnstructuredMesh, CartesianMesh((3, 3), (3.0, 3.0)))
        @test cells_inside_bounding_box(tm, [0.5, 0.5], [1.5, 1.5]) == [1, 2, 4, 5]
        @test cells_inside_bounding_box(tm, [-1.0, -1.0], [4.0, 4.0]) == 1:9
        @test cells_inside_bounding_box(tm, [0.1, 0.1], [0.2, 0.2]) == [1]
    end
end

@testset "RadialMeshes" begin
    @testset "spiral_mesh" begin
        spacing = [0.0, 0.5, 1.0]
        nrotations = 5
        n_angular_sections = 10
        rmesh = Jutul.RadialMeshes.spiral_mesh(n_angular_sections, nrotations, spacing = spacing)
        geo = tpfv_geometry(rmesh)

        @testset "interior normals" begin
            for f in 1:number_of_faces(rmesh)
                l, r = rmesh.faces.neighbors[f]
                cl = geo.cell_centroids[:, l]
                cr = geo.cell_centroids[:, r]
                N = geo.normals[:, f]
                @test dot(N, cr - cl) > 0
            end
        end
        ##
        @testset "exterior normals" begin
            for f in 1:number_of_boundary_faces(rmesh)
                c = rmesh.boundary_faces.neighbors[f]
                cc = geo.cell_centroids[:, c]
                fc = geo.boundary_centroids[:, f]
                N = geo.boundary_normals[:, f]
                @test dot(N, fc - cc) > 0
            end
        end
    end
    @testset "radial mesh" begin
        nangle = 10
        radii = [0.2, 0.5, 1.0]
        for centerpoint in [true, false]
            m = Jutul.RadialMeshes.radial_mesh(nangle, radii; centerpoint = centerpoint)

            geo = tpfv_geometry(m)
            @testset "interior normals" begin
                for f in 1:number_of_faces(m)
                    l, r = m.faces.neighbors[f]
                    cl = geo.cell_centroids[:, l]
                    cr = geo.cell_centroids[:, r]
                    N = geo.normals[:, f]
                    @test dot(N, cr - cl) > 0
                end
            end
            @testset "exterior normals" begin
                for f in 1:number_of_boundary_faces(m)
                    c = m.boundary_faces.neighbors[f]
                    cc = geo.cell_centroids[:, c]
                    fc = geo.boundary_centroids[:, f]
                    N = geo.boundary_normals[:, f]
                    @test dot(N, fc - cc) > 0
                end
            end
        end
    end
end

@testset "extrude_mesh" begin
    m2d = UnstructuredMesh(CartesianMesh((2, 2)))
    set_mesh_entity_tag!(m2d, Cells(), :test_tag, :tag1, [1, 3])
    set_mesh_entity_tag!(m2d, Cells(), :test_tag, :tag2, [2, 4])
    m3d = Jutul.extrude_mesh(m2d, 2)
    geo = tpfv_geometry(m3d)

    @testset "volumes" begin
        for v in geo.volumes
            @test v ≈ 0.5^3
        end
    end
    @testset "interior normals" begin
        for f in 1:number_of_faces(m3d)
            l, r = m3d.faces.neighbors[f]
            cl = geo.cell_centroids[:, l]
            cr = geo.cell_centroids[:, r]
            N = geo.normals[:, f]
            @test dot(N, cr - cl) > 0
        end
    end
    @testset "exterior normals" begin
        for f in 1:number_of_boundary_faces(m3d)
            c = m3d.boundary_faces.neighbors[f]
            cc = geo.cell_centroids[:, c]
            fc = geo.boundary_centroids[:, f]
            N = geo.boundary_normals[:, f]
            @test dot(N, fc - cc) > 0
        end
    end
    @testset "num_faces" begin
        @test number_of_faces(m3d) == 12
        @test number_of_boundary_faces(m3d) == 24
    end

    @testset "num_cells" begin
        @test number_of_cells(m3d) == 8
    end
    @testset "faces per cell" begin
        for cell in 1:number_of_cells(m3d)
            @test length(m3d.faces.cells_to_faces[cell]) == 3
            @test length(m3d.boundary_faces.cells_to_faces[cell]) == 3
        end
    end
    @testset "cell tags" begin
        @test get_mesh_entity_tag(m3d, Cells(), :test_tag, :tag1) == [1, 3, 5, 7]
        @test get_mesh_entity_tag(m3d, Cells(), :test_tag, :tag2) == [2, 4, 6, 8]
    end
end

@testset "refine_mesh" begin
    function check_normals(m_refined)
        geo = tpfv_geometry(m_refined)
        for f in 1:number_of_faces(m_refined)
            l, r = m_refined.faces.neighbors[f]
            cl = geo.cell_centroids[:, l]
            cr = geo.cell_centroids[:, r]
            N = geo.normals[:, f]
            @test dot(N, cr - cl) > 0
        end
        for f in 1:number_of_boundary_faces(m_refined)
            c = m_refined.boundary_faces.neighbors[f]
            cc = geo.cell_centroids[:, c]
            fc = geo.boundary_centroids[:, f]
            N = geo.boundary_normals[:, f]
            @test dot(N, fc - cc) > 0
        end
    end

    @testset "2D refinement" begin
        @testset "refine all cells" begin
            m = UnstructuredMesh(CartesianMesh((2, 2)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, [1, 2, 3, 4]; extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            @test number_of_cells(m2) == 16
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            @test length(result.cell_map) == number_of_cells(m2)
            check_normals(m2)
        end
        @testset "refine single cell" begin
            m = UnstructuredMesh(CartesianMesh((3, 3)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, [5]; extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            @test number_of_cells(m2) == 12
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            @test count(==(5), result.cell_map) == 4
            check_normals(m2)
        end
        @testset "refine with variable deltas" begin
            m = UnstructuredMesh(CartesianMesh((3, 2), ([1.0, 3.0, 4.0], [1.0, 2.0])))
            geo_orig = tpfv_geometry(m)
            m2 = refine_mesh(m, [1, 6])
            geo = tpfv_geometry(m2)
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            check_normals(m2)
        end
    end

    @testset "3D refinement" begin
        @testset "refine all cells" begin
            m = UnstructuredMesh(CartesianMesh((2, 2, 2)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, 1:8; extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            @test number_of_cells(m2) == 64
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            @test length(result.cell_map) == 64
            check_normals(m2)
        end
        @testset "refine single cell" begin
            m = UnstructuredMesh(CartesianMesh((2, 2, 2)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, [1]; extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            @test number_of_cells(m2) == 15
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            @test count(==(1), result.cell_map) == 8
            check_normals(m2)
        end
        @testset "refine with variable deltas" begin
            m = UnstructuredMesh(CartesianMesh((3, 2, 2), ((1.0, 3.0, 4.0), (1.0, 2.0), 1.0)))
            geo_orig = tpfv_geometry(m)
            m2 = refine_mesh(m, [1, 12])
            geo = tpfv_geometry(m2)
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            check_normals(m2)
        end
    end

    @testset "cell_map output" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        result = refine_mesh(m, [1, 3]; extra_out = true)
        cell_map = result.cell_map
        # Cells 1 and 3 each produce 4 sub-cells, cells 2 and 4 stay
        @test length(cell_map) == number_of_cells(result.mesh)
        @test count(==(1), cell_map) == 4
        @test count(==(2), cell_map) == 1
        @test count(==(3), cell_map) == 4
        @test count(==(4), cell_map) == 1
    end

    @testset "factor = 1 (no-op)" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        m2 = refine_mesh(m, [1, 2]; factor = 1)
        @test number_of_cells(m2) == number_of_cells(m)
    end

    @testset "per-cell factors" begin
        m = UnstructuredMesh(CartesianMesh((3, 3)))
        geo_orig = tpfv_geometry(m)
        m2 = refine_mesh(m, [1, 5, 9]; factor = [2, 1, 2])
        geo = tpfv_geometry(m2)
        # Cell 1 and 9 refined (4 sub each), cell 5 stays = 4 + 1 + 4 + 6 unrefined = 15
        @test number_of_cells(m2) == 15
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_normals(m2)
    end
end
