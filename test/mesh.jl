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

using MAT
@testset "UnstructuredMesh" begin
    fn = joinpath(pathof(Jutul), "..", "..", "data", "testgrids", "pico.mat")
    exported = MAT.matread(fn)
    G_raw = exported["G"]
    g = MRSTWrapMesh(G_raw)
    G = UnstructuredMesh(g)
    @testset "conversion" begin
        G2 = UnstructuredMesh(g)
        @test G2 isa UnstructuredMesh
    end
    @testset "basics" begin
        function test_faces(G, g)
            for i = 1:number_of_faces(G)
                f_ix = G.face_index[i]
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
end
