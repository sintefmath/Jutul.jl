using Jutul

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
