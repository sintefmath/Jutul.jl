using Jutul, Test

@testset "Original cell indices survive repeated submesh extraction" begin
    original = UnstructuredMesh(CartesianMesh((4,3,2),(4.0,3.0,2.0)))
    selected = [12,1,6,24,19]
    active = extract_submesh(original,selected)
    @test collect(active.cell_map) == selected
    @test number_of_cells(active) == length(selected)
    @test [Jutul.cell_ijk(active,i) for i in 1:5] ==
          [Jutul.cell_ijk(original,i) for i in selected]
    order = [4,1,3]
    nested = extract_submesh(active,order)
    @test collect(nested.cell_map) == selected[order]
    @test [Jutul.cell_ijk(nested,i) for i in 1:3] ==
          [Jutul.cell_ijk(original,i) for i in selected[order]]
    again = extract_submesh(nested,[3,1])
    @test collect(again.cell_map) == selected[order[[3,1]]]
    @test collect(active.cell_map) == selected # Parent mapping remains intact.

    # Also cover meshes constructed without an original-cell map.
    plain = UnstructuredMesh(original.faces.cells_to_faces,
        original.boundary_faces.cells_to_faces,original.faces.faces_to_nodes,
        original.boundary_faces.faces_to_nodes,original.node_points,
        original.faces.neighbors,original.boundary_faces.neighbors;
        structure=original.structure)
    @test isnothing(plain.cell_map)
    sub = extract_submesh(plain,selected)
    @test collect(sub.cell_map) == selected
    @test collect(extract_submesh(sub,order).cell_map) == selected[order]
end

@testset "Face positions for meshes without internal faces" begin
    for T in (Int32,Int64), nc in (0,1,2,5)
        faces,positions = Jutul.get_facepos(zeros(T,2,0),nc)
        @test isempty(faces)
        @test eltype(faces) == T
        @test positions == ones(T,nc+1)
        @test eltype(positions) == T
    end
    for args in ((),(nothing,))
        faces,positions = Jutul.get_facepos(zeros(Int,2,0),args...)
        @test isempty(faces)
        @test positions == [1,1] # Preserve the legacy unspecified-cell default.
    end
    grid = UnstructuredMesh(CartesianMesh((5,1,1),(5.0,1.0,1.0)))
    isolated = extract_submesh(grid,[5,1,3])
    @test number_of_faces(isolated) == 0
    @test number_of_cells(isolated) == 3
    faces,positions = Jutul.get_facepos(zeros(Int,2,0),number_of_cells(isolated))
    @test isempty(faces)
    @test positions == [1,1,1,1]
    # Exercise the actual downstream assembly bookkeeping, not only the helper.
    assembly = Jutul.FastAssemblyData(isolated)
    @test length(assembly.cells_to_faces) == 3
    @test all(length(assembly.cells_to_faces[c]) == 6 for c in 1:3)
    @test Jutul.get_facepos([1 2;2 3],3) == ([1,1,2,2],[1,2,4,5])
end
