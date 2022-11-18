using Jutul, Test

@testset "entity_eachindex" begin
    N = 3
    M = 5
    @test entity_eachindex(ones(N, M)) == Base.OneTo(M)
    @test entity_eachindex(ones(M)) == Base.OneTo(M)
end

@testset "load_balanced_endpoint" begin
    for n in 1:100
        for m in 1:n
            # Test endpoints
            @test Jutul.load_balanced_endpoint(0, n, m) == 0
            @test Jutul.load_balanced_endpoint(m, n, m) == n
            count = 0
            for i in 1:m
                # Test each block
                start = Jutul.load_balanced_endpoint(i-1, n, m)
                stop = Jutul.load_balanced_endpoint(i, n, m)
                @test start < stop
                delta = stop - start
                @test delta == floor(n/m) || delta == ceil(n/m)
                count += delta
            end
            # Check that the interval was partitioned
            @test count == n
        end
    end
end
