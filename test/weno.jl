using Jutul, Test
@testset "WENO setup" begin
    mesh_2d = UnstructuredMesh(CartesianMesh((3, 3)))
    @testset "2D" begin
        call = Jutul.WENO.weno_discretize_cells(DataDomain(mesh_2d))
        ix = 5
        c = call[ix]
        pset = c.planar_set
        cells = map(x -> x.cell, c.stencil)

        @test length(pset) == 4
        for quad in pset
            @test length(quad) == 3
            i, j, k = quad
            @test cells[i] == ix
            # Neighbor set in 2D should TPFA pairs of 2, 4, 6, 8
            c1 = cells[j]
            c2 = cells[k]
            if c1 > c2
                c2, c1 = c1, c2
            end
            @test (c1, c2) in [(2, 4), (4, 8), (6, 8), (2, 6)]
        end
    end
    mesh_3d = UnstructuredMesh(CartesianMesh((3, 3, 3)))

    w2d = Jutul.WENO.weno_discretize(DataDomain(mesh_2d))
    @test length(w2d) == number_of_faces(mesh_2d)
    @test eltype(w2d) == Jutul.WENO.WENOFaceDiscretization{2, 3, Float64}
    w3d = Jutul.WENO.weno_discretize(DataDomain(mesh_3d))
    @test length(w3d) == number_of_faces(mesh_3d)
    @test eltype(w3d) == Jutul.WENO.WENOFaceDiscretization{3, 4, Float64}
end
