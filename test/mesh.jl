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

function check_mesh_normals(m_refined)
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

@testset "refine_mesh" begin

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
            check_mesh_normals(m2)
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
            check_mesh_normals(m2)
        end
        @testset "refine with variable deltas" begin
            m = UnstructuredMesh(CartesianMesh((3, 2), ([1.0, 3.0, 4.0], [1.0, 2.0])))
            geo_orig = tpfv_geometry(m)
            m2 = refine_mesh(m, [1, 6])
            geo = tpfv_geometry(m2)
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            check_mesh_normals(m2)
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
            check_mesh_normals(m2)
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
            check_mesh_normals(m2)
        end
        @testset "refine with variable deltas" begin
            m = UnstructuredMesh(CartesianMesh((3, 2, 2), ((1.0, 3.0, 4.0), (1.0, 2.0), 1.0)))
            geo_orig = tpfv_geometry(m)
            m2 = refine_mesh(m, [1, 12])
            geo = tpfv_geometry(m2)
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            check_mesh_normals(m2)
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
        check_mesh_normals(m2)
    end

    @testset "higher factors by iteration" begin
        @testset "factor=4 all cells 2D" begin
            m = UnstructuredMesh(CartesianMesh((2, 2)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, [1, 2, 3, 4]; factor = 4, extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            # factor=4 = 2 passes of factor-2: 4 cells × 4^2 = 64
            @test number_of_cells(m2) == 64
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            @test length(result.cell_map) == 64
            @test all(c -> 1 <= c <= 4, result.cell_map)
            check_mesh_normals(m2)
        end
        @testset "factor=4 single cell 2D" begin
            m = UnstructuredMesh(CartesianMesh((3, 3)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, [5]; factor = 4, extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            @test count(==(5), result.cell_map) == 16
            check_mesh_normals(m2)
        end
        @testset "factor=3 2D (rounds up to 2 passes)" begin
            m = UnstructuredMesh(CartesianMesh((2, 2)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, [1]; factor = 3, extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            # ceil(log2(3)) = 2 passes → 16 sub-cells for the refined cell
            @test count(==(1), result.cell_map) == 16
            check_mesh_normals(m2)
        end
        @testset "factor=4 single cell 3D" begin
            m = UnstructuredMesh(CartesianMesh((2, 2, 2)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, [1]; factor = 4, extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            @test count(==(1), result.cell_map) == 64
            check_mesh_normals(m2)
        end
        @testset "per-cell mixed high factors" begin
            m = UnstructuredMesh(CartesianMesh((3, 3)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, [1, 5, 9]; factor = [4, 2, 1], extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            @test count(==(1), result.cell_map) == 16
            @test count(==(5), result.cell_map) == 4
            @test count(==(9), result.cell_map) == 1
            check_mesh_normals(m2)
        end
    end

    @testset "tuple factor" begin
        @testset "2D tuple factor" begin
            m = UnstructuredMesh(CartesianMesh((2, 2)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, [1, 2, 3, 4]; factor = (2, 2), extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            @test number_of_cells(m2) == 16
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            check_mesh_normals(m2)
        end
        @testset "2D tuple factor (2,1)" begin
            m = UnstructuredMesh(CartesianMesh((2, 2)))
            geo_orig = tpfv_geometry(m)
            m2 = refine_mesh(m, [1, 2, 3, 4]; factor = (2, 1))
            geo = tpfv_geometry(m2)
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            check_mesh_normals(m2)
        end
        @testset "3D tuple factor (2,2,1)" begin
            m = UnstructuredMesh(CartesianMesh((2, 2, 2)))
            geo_orig = tpfv_geometry(m)
            result = refine_mesh(m, 1:8; factor = (2, 2, 1), extra_out = true)
            m2 = result.mesh
            geo = tpfv_geometry(m2)
            @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
            check_mesh_normals(m2)
        end
        @testset "tuple factor (1,1) is no-op" begin
            m = UnstructuredMesh(CartesianMesh((2, 2)))
            m2 = refine_mesh(m, [1, 2]; factor = (1, 1))
            @test number_of_cells(m2) == number_of_cells(m)
        end
        @testset "tuple dimension mismatch" begin
            m = UnstructuredMesh(CartesianMesh((2, 2)))
            @test_throws AssertionError refine_mesh(m, [1]; factor = (2, 2, 1))
        end
    end

    @testset "cell_map through iterations" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        result = refine_mesh(m, [1, 3]; factor = 4, extra_out = true)
        cell_map = result.cell_map
        @test length(cell_map) == number_of_cells(result.mesh)
        # cell 1: 16 sub-cells, cell 3: 16 sub-cells, cells 2 and 4: 1 each
        @test count(==(1), cell_map) == 16
        @test count(==(2), cell_map) == 1
        @test count(==(3), cell_map) == 16
        @test count(==(4), cell_map) == 1
    end
end

@testset "refine_mesh_radial" begin

    @testset "default sectors (no edge splitting)" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        geo_orig = tpfv_geometry(m)
        result = refine_mesh_radial(m, [1, 2, 3, 4]; extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        @test number_of_cells(m2) == 16
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        @test length(result.cell_map) == 16
        check_mesh_normals(m2)
    end

    @testset "single cell" begin
        m = UnstructuredMesh(CartesianMesh((3, 3)))
        geo_orig = tpfv_geometry(m)
        result = refine_mesh_radial(m, [5]; extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        @test number_of_cells(m2) == 12
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        @test count(==(5), result.cell_map) == 4
        check_mesh_normals(m2)
    end

    @testset "n_sectors = 8 (with edge splitting)" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        geo_orig = tpfv_geometry(m)
        result = refine_mesh_radial(m, [1]; n_sectors = 8, extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        @test count(==(1), result.cell_map) == 8
        check_mesh_normals(m2)
    end

    @testset "n_sectors = 3 (fewer than edges)" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        geo_orig = tpfv_geometry(m)
        m2 = refine_mesh_radial(m, [1, 2, 3, 4]; n_sectors = 3)
        geo = tpfv_geometry(m2)
        # 4 edges > 3 sectors, so 4 sectors per cell
        @test number_of_cells(m2) == 16
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "empty cells" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        m2 = refine_mesh_radial(m, Int[])
        @test number_of_cells(m2) == 4
    end

    @testset "variable deltas" begin
        m = UnstructuredMesh(CartesianMesh((3, 2), ([1.0, 3.0, 4.0], [1.0, 2.0])))
        geo_orig = tpfv_geometry(m)
        m2 = refine_mesh_radial(m, [1, 6])
        geo = tpfv_geometry(m2)
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "n_rings = 2" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        geo_orig = tpfv_geometry(m)
        result = refine_mesh_radial(m, [1]; n_rings = 2, extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        # 4 sectors × 2 rings = 8 sub-cells + 3 unrefined = 11
        @test number_of_cells(m2) == 11
        @test count(==(1), result.cell_map) == 8
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "n_rings = 2 all cells" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        geo_orig = tpfv_geometry(m)
        m2 = refine_mesh_radial(m, [1, 2, 3, 4]; n_rings = 2)
        geo = tpfv_geometry(m2)
        @test number_of_cells(m2) == 32
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "n_rings = 3" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        geo_orig = tpfv_geometry(m)
        result = refine_mesh_radial(m, [1]; n_rings = 3, extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        # 4 sectors × 3 rings = 12 + 3 unrefined = 15
        @test number_of_cells(m2) == 15
        @test count(==(1), result.cell_map) == 12
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "center_cell with n_rings = 2" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        geo_orig = tpfv_geometry(m)
        result = refine_mesh_radial(m, [1]; n_rings = 2, center_cell = true, extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        # 4 sectors × 1 ring + 1 center = 5 + 3 unrefined = 8
        @test number_of_cells(m2) == 8
        @test count(==(1), result.cell_map) == 5
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "center_cell with n_rings = 3" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        geo_orig = tpfv_geometry(m)
        result = refine_mesh_radial(m, [1]; n_rings = 3, center_cell = true, extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        # 4 sectors × 2 rings + 1 center = 9 + 3 = 12
        @test number_of_cells(m2) == 12
        @test count(==(1), result.cell_map) == 9
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "center_cell all cells" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        geo_orig = tpfv_geometry(m)
        m2 = refine_mesh_radial(m, [1, 2, 3, 4]; n_rings = 2, center_cell = true)
        geo = tpfv_geometry(m2)
        # 4 cells × (4 sectors × 1 ring + 1 center) = 20
        @test number_of_cells(m2) == 20
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "n_sectors = 8 with n_rings = 2 and center_cell" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        geo_orig = tpfv_geometry(m)
        result = refine_mesh_radial(m, [1]; n_sectors = 8, n_rings = 2, center_cell = true, extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        # 8 sectors × 1 ring + 1 center = 9 + 3 = 12
        @test number_of_cells(m2) == 12
        @test count(==(1), result.cell_map) == 9
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "variable deltas with n_rings" begin
        m = UnstructuredMesh(CartesianMesh((3, 2), ([1.0, 3.0, 4.0], [1.0, 2.0])))
        geo_orig = tpfv_geometry(m)
        m2 = refine_mesh_radial(m, [1, 6]; n_rings = 2)
        geo = tpfv_geometry(m2)
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "variable deltas with center_cell" begin
        m = UnstructuredMesh(CartesianMesh((3, 2), ([1.0, 3.0, 4.0], [1.0, 2.0])))
        geo_orig = tpfv_geometry(m)
        m2 = refine_mesh_radial(m, [1, 6]; n_rings = 2, center_cell = true)
        geo = tpfv_geometry(m2)
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end
end

@testset "merge_cells" begin

    @testset "merge two adjacent cells in 1D" begin
        m = UnstructuredMesh(CartesianMesh((3, 1)))
        geo_orig = tpfv_geometry(m)
        result = merge_cells(m, [[1, 2]]; extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        @test number_of_cells(m2) == 2
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        @test result.cell_map[1] == [1, 2]
        @test result.cell_map[2] == [3]
        check_mesh_normals(m2)
    end

    @testset "merge all cells" begin
        m = UnstructuredMesh(CartesianMesh((2, 1)))
        geo_orig = tpfv_geometry(m)
        result = merge_cells(m, [[1, 2]]; extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        @test number_of_cells(m2) == 1
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "merge multiple groups" begin
        m = UnstructuredMesh(CartesianMesh((4, 1)))
        geo_orig = tpfv_geometry(m)
        result = merge_cells(m, [[1, 2], [3, 4]]; extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        @test number_of_cells(m2) == 2
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        @test result.cell_map[1] == [1, 2]
        @test result.cell_map[2] == [3, 4]
        check_mesh_normals(m2)
    end

    @testset "merge 2x2 block in 3x3" begin
        m = UnstructuredMesh(CartesianMesh((3, 3)))
        geo_orig = tpfv_geometry(m)
        result = merge_cells(m, [[1, 2, 4, 5]]; extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        @test number_of_cells(m2) == 6
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "merge in 3D" begin
        m = UnstructuredMesh(CartesianMesh((2, 2, 2)))
        geo_orig = tpfv_geometry(m)
        result = merge_cells(m, [[1, 2]]; extra_out = true)
        m2 = result.mesh
        geo = tpfv_geometry(m2)
        @test number_of_cells(m2) == 7
        @test sum(geo_orig.volumes) ≈ sum(geo.volumes)
        check_mesh_normals(m2)
    end

    @testset "empty merge" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        m2 = merge_cells(m, Vector{Int}[])
        @test number_of_cells(m2) == 4
    end

    @testset "cell in multiple groups" begin
        m = UnstructuredMesh(CartesianMesh((3, 1)))
        @test_throws AssertionError merge_cells(m, [[1, 2], [2, 3]])
    end

    @testset "cell_map correctness" begin
        m = UnstructuredMesh(CartesianMesh((4, 1)))
        result = merge_cells(m, [[2, 3]]; extra_out = true)
        cm = result.cell_map
        @test length(cm) == 3
        @test cm[1] == [2, 3]
        @test cm[2] == [1]
        @test cm[3] == [4]
    end

    @testset "MeshUtils module access" begin
        m = UnstructuredMesh(CartesianMesh((2, 2)))
        m2 = Jutul.MeshUtils.merge_cells(m, [[1, 2]])
        @test number_of_cells(m2) == 3
    end
end
