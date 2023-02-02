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
##
@testset "jutul_config" begin
    cfg = JutulConfig("Test configuration")
    add_option!(cfg, :abc, 3.0, "My test number")
    add_option!(cfg, :option_2, NaN, "A second option", description = "This option has a very long description to expand on what exactly it entails to be the second option")
    add_option!(cfg, :limited_value, 3, "Limited value", valid_values = [1, 5, 3])
    add_option!(cfg, :limited_type, 3, "Limited type", valid_types = Float64)

    # Test that options cannot be overriden
    @test_throws "cannot replace" add_option!(cfg, :abc, 5)
    # Unless explicitly allowed
    add_option!(cfg, :abc, 7.0, replace = true)
    # Test retrieval
    @test cfg[:abc] == 7.0
    @test cfg[:limited_value] == 3
    # Test setting
    cfg[:limited_value] = 5
    @test cfg[:limited_value] == 5
    # Test asserts
    @test_throws "limited_value must be one of " cfg[:limited_value] = 4
    @test cfg[:limited_value] == 5

    for (inner, outer) in zip(pairs(cfg), pairs(cfg.values))
        k_i, v_i = inner
        k_o, v_o = outer
        @test k_i === k_o
        @test v_i === v_o
    end
    for (k_i, k_o) in zip(keys(cfg), keys(cfg.values))
        @test k_i == k_o
    end
    for (v_i, v_o) in zip(values(cfg), values(cfg.values))
        # === due to NaN
        @test v_i === v_o
    end
    @test haskey(cfg, :abc)
    @test !haskey(cfg, :abcd)
end
