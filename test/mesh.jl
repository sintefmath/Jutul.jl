using Jutul
using Test

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
using Meshes
@testset "Meshes.jl interop" begin
    # This testing is a bit simple since it relies on the same ordering
    # in both Jutul.jl's cart grid and Meshes.jl
    dims = (2,2,3)
    grid  = CartesianGrid(dims)
    geo = tpfv_geometry(grid)
    jgrid = Jutul.CartesianMesh(dims, (2.0, 2.0, 3.0))
    jgeo = tpfv_geometry(jgrid)
    @test isapprox(jgeo.cell_centroids, geo.cell_centroids, atol = 1e-12)
    @test isapprox(jgeo.volumes, geo.volumes, atol = 1e-12)
end
##
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
        test_meshes = [
            CartesianMesh((3, 2, 2), ((1.0, 3.0, 4.0), (1.0, 2.0), 1.0)),
            CartesianMesh((3, 2, 2)),
            CartesianMesh((4, 1, 1)),
            CartesianMesh((9, 7, 5), origin = [0.2, 0.6, 10.1]),
            CartesianMesh((3, 2, 2), (10.0, 3.0, 5.0)),
            CartesianMesh((3, 2, 2), ([10.0, 5.0, π], 3.0, 5.0)),
            CartesianMesh((100, 3, 7))
            ]
        for g in test_meshes
            G = UnstructuredMesh(g)
            geo1 = tpfv_geometry(g)
            geo2 = tpfv_geometry(G)

            @testset "cells" begin
                @test geo1.volumes ≈ geo2.volumes
                @test geo1.cell_centroids ≈ geo2.cell_centroids
            end
            @testset "faces" begin
                @test geo1.neighbors == geo2.neighbors
                @test geo1.normals == geo2.normals
                @test geo1.areas ≈ geo2.areas
                @test geo1.face_centroids ≈ geo2.face_centroids
            end
            @testset "boundary" begin
                @test geo1.boundary_normals == geo2.boundary_normals
                @test geo1.boundary_neighbors == geo2.boundary_neighbors
                @test geo1.boundary_areas ≈ geo2.boundary_areas
                @test geo1.boundary_centroids ≈ geo2.boundary_centroids
            end
            @testset "half-faces" begin
                @test geo1.half_face_faces == geo2.half_face_faces
                @test geo1.half_face_cells == geo2.half_face_cells
            end
        end
        # 2D support missing
        @test_warn "Conversion from CartesianMesh to UnstructuredMesh is only fully supported for 3D grids. Converting 2D grid to 3D." UnstructuredMesh(CartesianMesh((3, 2)))
        # 1D support missing
        @test_warn "Conversion from CartesianMesh to UnstructuredMesh is only fully supported for 3D grids. Converting 1D grid to 3D." UnstructuredMesh(CartesianMesh((3,)))
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
