using Jutul, LinearAlgebra, SparseArrays, Test

@testset "CPU preconditioner updates and applications" begin
    @testset "ILU application coefficients" begin
        A = sparse([4.0 -1.0 0.0; -2.0 5.0 -1.0; 0.0 -3.0 6.0])
        b = [1.0, -2.0, 3.0]
        original = [3.0, 1.0, -2.0]
        for matrix in (A, Jutul.StaticSparsityMatrixCSR(sparse(transpose(A)); nthreads = 1))
            preconditioner = ILUZeroPreconditioner()
            Jutul.update_preconditioner!(preconditioner, matrix, b, nothing, nothing)
            solution = A \ b
            for (alpha, beta) in ((1.0, 0.0), (2.5, 0.0), (-1.0, 0.5), (0.0, -2.0))
                result = copy(original)
                @test Jutul.apply!(result, preconditioner, b, alpha, beta) === result
                @test result ≈ alpha*solution + beta*original
            end
        end
    end
    
    @testset "SPAI0 numeric updates replace normalization" begin
        A = sparse([4.0 -1.0 0.0; -2.0 5.0 -1.0; 0.0 -3.0 6.0])
        expected = diag(A) ./ vec(sum(abs2, A; dims = 2))
        for matrix in (copy(A), Jutul.StaticSparsityMatrixCSR(sparse(transpose(A)); nthreads = 1))
            preconditioner = SPAI0Preconditioner()
            for _ in 1:3
                Jutul.update_preconditioner!(preconditioner, matrix, ones(3), nothing, nothing)
                @test preconditioner.factor.D ≈ expected
            end
            nonzeros(matrix) .*= 2
            Jutul.update_preconditioner!(preconditioner, matrix, ones(3), nothing, nothing)
            @test preconditioner.factor.D ≈ expected/2
        end
    end
end
