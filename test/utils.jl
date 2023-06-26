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
    @test tuple(keys(d)...) == (
        :neighbors,
        :areas,
        :normals,
        :face_centroids,
        :cell_centroids,
        :volumes,
        :half_face_cells,
        :half_face_faces,
        :boundary_areas,
        :boundary_centroids,
        :boundary_normals,
        :boundary_neighbors,
        :cell_vector,
        :face_vector,
        :scalar,
        :data_2d,
        :data_3d
        )
end

@testset "get_1d_interpolator" begin
    x = collect(0:0.1:4)
    I = get_1d_interpolator(x, sin.(x))
    @test isapprox(I(π/2), 1.0, atol = 1e-2)
end

@testset "compress_timesteps" begin
    # Two test forces
    f1 = (f = 1, )
    f2 = (f = 2, )
    @test compress_timesteps([1.0, 2.0, 3.0]) == ([6.0], nothing)
    @test compress_timesteps([1.0, 2.0, 3.0], nothing) == ([6.0], nothing)
    @test compress_timesteps([1.0, 2.0, 3.0], f1) == ([6.0], f1)
    @test compress_timesteps([1.0, 2.0, 3.0, 4.0, 5.0], [f1, f1, f2, f2, f2]) == ([3.0, 12.0], [f1, f2])
    # Changing and merging forces
    @test compress_timesteps([1.0, 2.0, 3.0, 4.0, 5.0], [f2, f1, f2, f1, f1]) == ([1.0, 2.0, 3.0, 9.0], [f2, f1, f2, f1])
    # Limit to max time-step
    @test compress_timesteps([1.0, 3.0, 0.5, 1.0], max_step = 1.0) == ([1.0, 1.0, 1.0, 1.0, 1.0, 0.5], nothing)
    @test compress_timesteps([0.9, 3.0, 2.5, 0.62], [f1, f1, f2, f2], max_step = 1.0) == ([1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.12], [f1, f1, f1, f1, f2, f2, f2])
end

using Jutul, Test
@testset "Transmissibilities" begin
    test_meshes = [
        CartesianMesh((2, 2))
        CartesianMesh((3, 7, 1), (0.2, 0.3, 1.9))
        CartesianMesh((3, 4), ([0.1, 0.2, 0.5], [0.4, 0.1, 0.7, 0.2]))
        CartesianMesh((5,), (1000.0,))
    ]
    for g in test_meshes
        d = DataDomain(g)
        nc = number_of_cells(g)
        data = rand(nc)
        d[:data] = data
        # Test half trans
        T1 = compute_half_face_trans(d, data)
        T2 = compute_half_face_trans(d, :data)
        @test T1 == T2
        # Test full trans
        T1 = compute_face_trans(d, data)
        T2 = compute_face_trans(d, :data)
        @test T1 == T2
        # Test boundary trans
        Th1 = compute_boundary_trans(d, data)
        Th2 = compute_boundary_trans(d, :data)
        @test length(Th1) == number_of_boundary_faces(g)
        @test Th1 == Th2
    end
    # Boundary trans correctness
    g = CartesianMesh((2, 2, 2))
    d = DataDomain(g)
    d[:permeability] = 1.0
    Tb1 = compute_boundary_trans(d)
    @test all(Tb1 .== 1)
end
