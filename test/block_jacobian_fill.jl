using Jutul, Test

@testset "Block Jacobian filling" begin
    derivatives = reshape(collect(1.0:18.0), 3, 3, 2)
    for adjoint in (false, true)
        context = DefaultContext(; matrix_layout = Jutul.BlockMajorLayout(adjoint))
        cache = CompactAutoDiffCache(3, 2, 3; context)
        cache.jacobian_positions.first_positions .= [1 10]
        T = eltype(cache.entries)
        for cell in 1:2, equation in 1:3
            partials = Jutul.ForwardDiff.Partials(Tuple(derivatives[equation, :, cell]))
            cache.entries[equation, cell] = T(Float64(10cell + equation), partials)
        end

        jacobian, residual = zeros(18), zeros(6)
        Jutul.fill_equation_entries!(jacobian, residual, (; context), cache)
        expected = adjoint ? permutedims(derivatives, (2, 1, 3)) : derivatives
        @test jacobian == vec(expected)
        @test residual == [11, 21, 12, 22, 13, 23]

        blocks = zeros(Jutul.SMatrix{3, 3, Float64, 9}, 2)
        buffer = Jutul.jacobian_fill_buffer(vec(reinterpret(reshape, Float64, blocks)))
        for cell in 1:2
            positions = Jutul.jacobian_position_block(cache.jacobian_positions, cell, 3)
            Jutul.update_jacobian_block!(buffer, positions, Jutul.SMatrix{3, 3}(derivatives[:, :, cell]))
        end
        @test collect(buffer) == vec(expected)
    end
end
