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
@testset "JutulConfig" begin
    cfg = JutulConfig("Test configuration")
    add_option!(cfg, :abc, 3.0, "My test number")
    add_option!(cfg, :option_2, NaN, "A second option", description = "This option has a very long description to expand on what exactly it entails to be the second option")
    add_option!(cfg, :limited_value, 3, "Limited value", values = [1, 5, 3])
    add_option!(cfg, :limited_type, 3, "Limited type", types = Float64)

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

@testset "DataDomain" begin
    # number of faces should be distinct from cells
    nx = 2
    ny = 3
    g = CartesianMesh((nx, ny))
    n = number_of_cells(g)
    d = DataDomain(g)
    @test count_entities(d, Cells()) == nx*ny
    ## Setting entity
    v = rand(n)
    d[:cell_vector] = v
    # Set with default entity
    @test d[:cell_vector] == v
    @test d[:cell_vector, Cells()] == v
    @test_throws "Expected property cell_vector to be defined for Faces(), but was stored as Cells()" d[:cell_vector, Faces()]
    # Set with face entity
    nf = number_of_faces(g)
    vf = rand(nf)
    d[:face_vector, Faces()] = vf
    @test d[:face_vector] == vf
    @test d[:face_vector, Faces()] == vf
    # Overwrite
    d[:cell_vector] = rand(n)
    # Overwrite with same entity
    d[:cell_vector, Cells()] = rand(n)
    ##
    d[:scalar] = 1.0
    # Should be repeated onto all cells
    @test d[:scalar] == ones(n)
    # Test non-entity values
    m_big = rand(93, 2)
    d[:scalar, nothing] = m_big
    @test d[:scalar] == m_big
    @test d[:scalar, NoEntity()] == m_big
    # 2D is ok
    d2d = rand(10, n)
    d[:data_2d] = d2d
    @test d[:data_2d] == d2d
    @test_throws "" d[:data_2d] = rand(n, 10)
    # 3d, 4d, ... is ok
    d3d = rand(10, 20, n)
    d[:data_3d] = d3d
    @test d[:data_3d] == d3d
    @test_throws "AssertionError: Number of columns for Matrix scalar defined on Cells() should be 6, was 2" d[:scalar] = rand(93, 2)
    @test tuple(keys(d)...) == (:neighbors, :areas, :normals, :face_centroids, :cell_centroids, :volumes, :cell_vector, :face_vector, :scalar, :data_2d, :data_3d)
end

@testset "get_1d_interpolator" begin
    x = collect(0:0.1:4)
    I = get_1d_interpolator(x, sin.(x))
    @test isapprox(I(π/2), 1.0, atol = 1e-2)
end
