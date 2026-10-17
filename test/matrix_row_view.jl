using Jutul, LinearAlgebra, Test

@testset "Matrix rows preserve layout and aliases" begin
    A = [1.0 2.0 3.0; 4.0 5.0 6.0]
    @test Jutul.matrix_row_view(A, 2) == [4, 5, 6]
    for wrapper in (transpose, adjoint)
        row = Jutul.matrix_row_view(wrapper(A), 2)
        @test row == [2, 5]
        @test parent(row) === A
        row[1] = 7
        @test A[1, 2] == 7
        A[1, 2] = 2
    end
    B = complex.(A, 1)
    @test Jutul.matrix_row_view(B', 1) == conj.(B[:, 1])
end
