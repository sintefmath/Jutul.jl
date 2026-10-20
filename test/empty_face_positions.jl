using Jutul, Test

@testset "Face positions for meshes without internal faces" begin
    for T in (Int32, Int64), nc in (0, 1, 2, 5)
        faces, positions = Jutul.get_facepos(zeros(T, 2, 0), nc)
        @test isempty(faces)
        @test eltype(faces) == T
        @test positions == ones(T, nc + 1)
        @test eltype(positions) == T
    end
    for args in ((), (nothing,))
        faces, positions = Jutul.get_facepos(zeros(Int, 2, 0), args...)
        @test isempty(faces)
        # Preserve the legacy unspecified-cell default.
        @test positions == [1, 1]
    end
    grid = UnstructuredMesh(CartesianMesh((5, 1, 1), (5.0, 1.0, 1.0)))
    isolated = extract_submesh(grid, [5, 1, 3])
    @test number_of_faces(isolated) == 0
    @test number_of_cells(isolated) == 3
    faces, positions = Jutul.get_facepos(zeros(Int, 2, 0), number_of_cells(isolated))
    @test isempty(faces)
    @test positions == [1, 1, 1, 1]
    # Exercise downstream assembly bookkeeping, not only the helper.
    assembly = Jutul.FastAssemblyData(isolated)
    @test length(assembly.cells_to_faces) == 3
    @test all(length(assembly.cells_to_faces[c]) == 6 for c in 1:3)
    @test Jutul.get_facepos([1 2; 2 3], 3) == ([1, 1, 2, 2], [1, 2, 4, 5])
end
