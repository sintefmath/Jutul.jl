using Test
using Jutul
using StaticArrays
import Jutul.NFVM as NFVM

@testset "triplets" begin
    u_val = π
    factor = 9.13172
    offsets = [0.0, 100.0, 1e6]
    n = 10

    # TODO: Test offsets.
    @testset "2D" begin
        T = SVector{2, Float64}
        x_t = T(0.0, 0.0)
        l = T(u_val, 0.0)
        all_x = [
            factor.*T(-1.0, 0.0),
            factor.*T(0.0, -1.0),
            factor.*T(1.0, 0.0),
            factor.*T(0.0, 1.0),
        ]
        pair, w = NFVM.find_minimizing_basis(x_t, l, all_x)
        l_r = NFVM.reconstruct_l(pair, w, x_t, all_x)
        # Should be the same.
        @test l_r ≈ l

        ix = findfirst(isequal(3), pair)
        @test !isnothing(ix)
        @test w[ix]/norm(w, 2) ≈ 1.0
        for offset in offsets
            for xi in range(0.0, 1.0, n)
                for yi in range(0.0, 1.0, n)
                    if xi == yi == 0.0
                        continue
                    end
                    l = u_val.*T(xi, yi)
                    x_t_i = x_t .+ offset
                    x = copy(all_x)
                    for i in eachindex(x)
                        x[i] = x[i] .+ offset
                    end
                    pair, w = NFVM.find_minimizing_basis(x_t_i, l, x)
                    l_r = NFVM.reconstruct_l(pair, w, x_t_i, x)
                    @test l_r ≈ l
                end
            end
        end
    end
    @testset "3D" begin
        T = SVector{3, Float64}
        x_t = T(0.0, 0.0, 0.0)
        l = T(u_val, 0.0, 0.0)
        all_x = [
            factor.*T(-1.0, 0.0, 0.0),
            factor.*T(0.0, -1.0, 0.0),
            factor.*T(0.0, 0.0, -1.0),
            factor.*T(1.0, 0.0, 0.0),
            factor.*T(0.0, 1.0, 0.0),
            factor.*T(0.0, 0.0, 1.0)
        ]
        triplet, w = NFVM.find_minimizing_basis(x_t, l, all_x)

        l_r = NFVM.reconstruct_l(triplet, w, x_t, all_x)
        @test l_r ≈ l

        ix = findfirst(isequal(4), triplet)
        @test !isnothing(ix)
        @test w[ix]/norm(w, 2) ≈ 1.0
        for offset in offsets
            for xi in range(0.0, 1.0, n)
                for yi in range(0.0, 1.0, n)
                    for zi in range(0.0, 1.0, n)
                        if xi == yi == zi == 0.0
                            continue
                        end
                        l = u_val.*T(xi, yi, zi)
                        x_t_i = x_t .+ offset
                        x = copy(all_x)
                        for i in eachindex(x)
                            x[i] = x[i] .+ offset
                        end
                        trip, w = NFVM.find_minimizing_basis(x_t_i, l, x)
                        l_r = NFVM.reconstruct_l(trip, w, x_t_i, x)
                        @test l_r ≈ l
                    end
                end
            end
        end
        @testset "special cases" begin
            Tt = SVector{2, Float64}
            x_t = Tt(0.16666666666666666, 0.8333333333333334)
            points = [
                Tt(0.3333333333333333, 0.8333333333333335),
                Tt(0.16666666666666663, 1.666666666666667),
                Tt(0.0, 0.8333333333333334),
                Tt(0.16666666666666666, 0.0)
            ]
            l = Tt(0.0, 0.0010471975511965976)
            triplet, w = NFVM.find_minimizing_basis(x_t, l, points, verbose = false)
            l_r = NFVM.reconstruct_l(triplet, w, x_t, points)
            @test l_r ≈ l

            Tv = SVector{2, Float64}
            x_t = Tv(0.16875801386133837, 0.8567381844756641)
            AKn = Tv(0.005227888394818919, -6.757589394032612e-6)
            points = [
                Tv(0.33463202673281195, 0.8540976239029396),
                Tv(0.16912337164668087, 1.689472036394667),
                Tv(0.003131953514408839, 0.855309095397525),
                Tv(0.16852587628122123, 0.021593752012451912)
            ]
            ijk, w = NFVM.find_minimizing_basis(x_t, AKn, points, verbose = true)
            @test NFVM.reconstruct_l(ijk, w, x_t, points) ≈ AKn
            @test ijk == (1, 2)
            @test w ≈ [0.03151702159776656, 9.182407443929852e-5]
        end
    end
end
