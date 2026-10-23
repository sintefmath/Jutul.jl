using Jutul, Test
using KernelAbstractions: CPU

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
        expected = if adjoint
            permutedims(derivatives, (2, 1, 3))
        else
            derivatives
        end
        @test jacobian == vec(expected)
        @test residual == [11, 21, 12, 22, 13, 23]

        blocks = zeros(Jutul.SMatrix{3, 3, Float64, 9}, 2)
        jac = Jutul.StaticSparsityMatrixCSR(blocks, [1, 2], [1, 2, 3], 2, 2, CPU())
        r, dx = zeros(6), zeros(6)
        system = LinearizedSystem(jac, r, dx, vec(reinterpret(reshape, Float64, blocks)), r, dx, matrix_layout(context))
        ka_context = KernelAbstractionsContext(CPU(); matrix_layout = matrix_layout(context))
        adapted = Jutul.Adapt.adapt(ka_context, system)
        buffer = adapted.jac_buffer
        @test parent(buffer) === Jutul.nonzeros(adapted.jac)
        for cell in 1:2
            positions = Jutul.jacobian_position_block(cache.jacobian_positions, cell, 3)
            Jutul.update_jacobian_block!(buffer, positions, Jutul.SMatrix{3, 3}(derivatives[:, :, cell]))
        end
        @test collect(buffer) == vec(expected)

        flux = CompactAutoDiffCache(3, 3, 3; context)
        flux.jacobian_positions.first_positions .= [19 0 28]
        flux.entries .= hcat(-0.5cache.entries[:, 1], 0.25cache.entries[:, 2], -0.5cache.entries[:, 1])
        jacobian, residual = zeros(36), zeros(3, 2)
        for cell in 1:2
            Jutul.fill_conservation_eq!(jacobian, residual, cell, cache, flux, [1, 3, 4], Val(3), Val(3))
        end
        d1, d2 = derivatives[:, :, 1], derivatives[:, :, 2]
        expected = cat(0.5d1 + 0.25d2, d2 - 0.5d1, 0.5d1, 0.5d1; dims = 3)
        if adjoint
            expected = permutedims(expected, (2, 1, 3))
        end
        @test jacobian == vec(expected)
        @test residual == [10.75 15.5; 11.5 16.0; 12.25 16.5]
    end
end
