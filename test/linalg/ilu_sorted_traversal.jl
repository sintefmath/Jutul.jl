using Jutul, Test
using Jutul.StaticArrays: SMatrix

@testset "ILU sorted traversal preserves general lookup arithmetic" begin
    for T in (Float32, Float64), Ti in (Int32, Int64), blocked in (false, true)
        block(v) = blocked ? SMatrix{2, 2, T, 4}(v, v, v, v) : v
        A = Jutul.StaticSparsityMatrixCSR(block.(T[-0.0, 4, Inf]),
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
    # Generic matrices do not require sorted targets.
    A = [1.0 2.0 3.0]
    columns = [3, 1, 2]
    expected, actual = zeros(3), zeros(3)
    Jutul.StaticCSR.process_partial_row!(expected, 1:3, columns, A, 1, 2.0)
    Jutul.StaticCSR.process_sorted_partial_row!(actual, 1:3, columns, A, 1, 2.0)
    @test isequal(actual, expected)
end
