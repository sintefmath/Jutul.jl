using Jutul
using LinearAlgebra
using SparseArrays
using Jutul.StaticArrays: SMatrix
using Test

@testset "CSR ILU scalar precision" begin
    for T in (Float32, Float64), Ti in (Int32, Int64)
        @testset "$T/$Ti" begin
            A = T[4 -1 0; -2 5 -1; 0 -3 6]
            csr = Jutul.StaticSparsityMatrixCSR(
                SparseMatrixCSC{T, Ti}(sparse(transpose(A))); nthreads = 1
            )
            b = T[1, -2, 3]
            x = zeros(T, 3)
            ilu = ILUZeroPreconditioner()
            Jutul.update_preconditioner!(ilu, csr, b, nothing, nothing)
            Jutul.apply!(x, ilu, b)
            @test A * x ≈ b
            @test eltype(x) === T
            @test Jutul.operator_nrows(ilu) == length(b)

            # Reuse the factorization after a numeric update.
            csr.nzval .*= T(2)
            Jutul.update_preconditioner!(ilu, csr, b, nothing, nothing)
            Jutul.apply!(x, ilu, b)
            @test (2A) * x ≈ b
        end
    end
end

@testset "CSR ILU block precision" begin
    for T in (Float32, Float64), Ti in (Int32, Int64)
        @testset "$T/$Ti" begin
            B = SMatrix{2, 2, T, 4}
            diagonal = B(4, -1, -2, 5)
            coupling = B(-1, 0, 0, -1)
            csr = Jutul.StaticSparsityMatrixCSR(
                [diagonal, coupling, coupling, diagonal],
                Ti[1, 2, 1, 2], Ti[1, 3, 5], 2, 2, nothing
            )
            factor = Jutul.StaticCSR.ilu0_csr(csr)
            b = T[1, -2, 3, -4]
            x = zeros(T, 4)
            Jutul.ilu_apply!(x, factor, b)
            A = [Matrix(diagonal) Matrix(coupling); Matrix(coupling) Matrix(diagonal)]
            @test A * x ≈ b
            @test eltype(x) === T
        end
    end
end
