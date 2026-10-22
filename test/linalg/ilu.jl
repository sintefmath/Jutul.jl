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

function test_sorted_ilu_partial_row(::Type{T}, ::Type{Ti}, ::Val{blocked}) where {T, Ti, blocked}
    block(v) = blocked ? SMatrix{2, 2, T, 4}(v, v, v, v) : v
    A = Jutul.StaticSparsityMatrixCSR(block.(T[-0.0, 4.0, Inf]),
        Ti[1, 1, 3], Ti[1, 4], 1, 4, nothing)
    columns = Ti[1, 1, 2, 3, 3, 4]
    for multiplier in T[-0.0, 0.0, 2.0, Inf, NaN], positions in (1:6, view(1:6, 2:5))
        expected = block.(T[-0.0, 0.0, 1.0, -Inf, NaN, 2.0])
        actual = copy(expected)
        Jutul.StaticCSR.process_partial_row!(expected, positions, columns, A, 1, block(multiplier))
        Jutul.StaticCSR.process_sorted_partial_row!(actual, positions, columns, A, 1, block(multiplier))
        @test isequal(actual, expected)
    end
end

@testset "CSR ILU sorted row lookup" begin
    for T in (Float32, Float64), Ti in (Int32, Int64), blocked in (Val(false), Val(true))
        test_sorted_ilu_partial_row(T, Ti, blocked)
    end
    # Other matrix types keep the original general lookup, including arbitrary
    # target order.
    A = [1.0 2.0 3.0]
    columns = [3, 1, 2]
    expected, actual = zeros(3), zeros(3)
    Jutul.StaticCSR.process_partial_row!(expected, 1:3, columns, A, 1, 2.0)
    Jutul.StaticCSR.process_sorted_partial_row!(actual, 1:3, columns, A, 1, 2.0)
    @test isequal(actual, expected)
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
