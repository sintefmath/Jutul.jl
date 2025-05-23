using Jutul, Test, StaticArrays, ForwardDiff

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
    for constant_dx in [true, false, missing]
        # Test scalar interpolation
        x = collect(0:0.1:4)
        I = get_1d_interpolator(x, sin.(x), constant_dx = constant_dx)
        @test isapprox(I(π/2), 1.0, atol = 1e-2)
        x = [0.0, 0.5, 1.0]
        Fx = x.^2

        F_approx = get_1d_interpolator(x, Fx, constant_dx = constant_dx)
        @test F_approx(0.5) ≈ 0.25
        @test F_approx(0.25) ≈ 0.25/2
        # Test block interpolation
        x1 = @SVector [0.0, 0.1]
        x2 = @SVector [0.25, 0.35]
        x3 = @SVector [1.0, 1.1]
        Fx_b = [x1, x2, x3]
        F_approx_b = get_1d_interpolator(x, Fx_b, constant_dx = constant_dx)
        @test F_approx_b(0.5) ≈ x2
        @test F_approx_b(0.25) ≈ (x1 + x2)/2
        # Check that default is to not extrapolate.
        @test F_approx_b(-1.0) ≈ x1
        @test F_approx_b(1.5) ≈ x3
    end
end

@testset "get_2d_interpolator" begin
    for constant_dx in [true, false, missing]
        for constant_dy in [true, false, missing]
            f(x, y) = sin(x) + cos(y) + 0.5*x
            xs = range(0.0, 4.0, 10)
            ys = range(0.0, 5.0, 8)
            xs = collect(xs)
            ys = collect(ys)
            nx = length(xs)
            ny = length(ys)
            fs = zeros(nx, ny)
            for (i, x) in enumerate(xs)
                for (j, y) in enumerate(ys)
                    fs[i, j] = f(x, y)
                end
            end
            I = get_2d_interpolator(xs, ys, fs,
                constant_dx = constant_dx, constant_dy = constant_dy)
            for (i, x) in enumerate(xs)
                for (j, y) in enumerate(ys)
                    @test I(x, y) ≈ f(x, y)
                end
            end
            nf = 100
            xfine = range(-1.0, 5.0, nf)
            yfine = range(-1.0, 6.0, nf)
            using ForwardDiff, Test
            F(X) = I(first(X), last(X))
            function F_num(X)
                ϵ = 1e-6
                x = X[1]
                y = X[2]
                v0 = I(x, y)
                vx = I(x + ϵ, y)
                vy = I(x, y + ϵ)
                dx = (vx - v0)/ϵ
                dy = (vy - v0)/ϵ
                return (dx, dy)
            end
            for (i, x) in enumerate(xfine)
                for (j, y) in enumerate(yfine)
                    dx_i, dy_i = ForwardDiff.gradient(F, [x, y])
                    dx_num, dy_num = F_num([x, y])
                    @test dx_num ≈ dx_i rtol=1e-3 atol=1e-8
                    @test dy_num ≈ dy_i rtol=1e-3 atol=1e-8
                end
            end
        end
    end
end

@testset "first_lower fast lookup" begin
    for nstep in 2:100
        for t in [(1.5, 3.9), (-100.0, 53.0), (-1e-3, 1e-3)]
            start, stop = t
            dx = collect(range(start, stop, length = nstep))
            lookup = Jutul.interpolation_constant_lookup(dx)
            @test lookup.dx ≈ (stop - start)/(nstep-1)
            wd = 0.1*(stop-start)
            for x in range(start - wd, stop + wd, length = 3*nstep)
                pos = Jutul.first_lower(dx, x)
                pos_l = Jutul.first_lower(dx, x, lookup)
                bnd = start + pos*lookup.dx
                bnd_l = start + pos_l*lookup.dx
                at_boundary = isapprox(x, bnd, atol = 1e-10) || isapprox(x, bnd_l, atol = 1e-10)
                @test pos == pos_l || at_boundary
            end
        end
    end
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

@testset "IndirectionMap" begin
    val = [1.0, 2.0, 3.0, 4.0, 5.0]
    ix = [1, 3, 6]

    m = Jutul.IndirectionMap(val, ix)

    @test m[2] == [3.0, 4.0, 5.0]
    @test m[1] == [1.0, 2.0]
    @test length(m) == 2
end

import Jutul: IndexRenumerator
@testset "IndexRenumerator" begin
    im = IndexRenumerator()
    @test im[3] == 1
    @test im[1] == 2
    @test im[7] == 3
    @test im[3] == 1
    @test !(4 in im)
    @test 3 in im
    @test 1 in im
    @test 7 in im
    @test Jutul.indices(im) == [3, 1, 7]
    @test Jutul.indices(IndexRenumerator([1, 5, 7, 2])) == [1, 5, 7, 2]
    @test Jutul.indices(IndexRenumerator([5, π, 3.0, 17.6])) == [5, π, 3.0, 17.6]
end

using Test, Jutul
import Jutul: numerical_type, numerical_eltype
@testset "numerical_type/eltype" begin
    num = 1.0
    ad = Jutul.get_ad_entity_scalar(1.0, 3, 1, tag = Cells())

    @test numerical_type(num) == Float64
    @test numerical_type(ad) == typeof(ad)
    @test numerical_type(NaN) == Float64
    tup = (a = 3, b = 4)
    @test_throws MethodError numerical_type(tup)

    @test numerical_eltype([ad, ad, ad]) == typeof(ad)
    @test numerical_eltype([ad ad; ad ad]) == typeof(ad)
    @test numerical_eltype([num, num, num]) == Float64
    @test numerical_eltype([num num; num num]) == Float64
end

@testset "jutul_output_path" begin
    @test isdir(jutul_output_path())
    @test last(splitdir(jutul_output_path("testname"))) == "testname"
end
@testset "mesh tags" begin
    for i in 1:2
        g = CartesianMesh((10,1,1))
        if i == 2
            g = UnstructuredMesh(g)
        end
        set_mesh_entity_tag!(g, Cells(), :group, :tag1, [1, 2, 3])
        set_mesh_entity_tag!(g, Cells(), :group, :tag2, [5, 4, 6])
        set_mesh_entity_tag!(g, Cells(), :group2, :tag1, [1, 2, 3])
        set_mesh_entity_tag!(g, Cells(), :group2, :tag1, [3, 7, 1])

        @test_throws "Tag value must not exceed 10 for Cells()" set_mesh_entity_tag!(g, Cells(), :group2, :tag1, [21])
        @test get_mesh_entity_tag(g, Cells(), :group, :tag1) == [1, 2, 3]
        @test get_mesh_entity_tag(g, Cells(), :group, :tag2) == [4, 5, 6] # Sorted.
        @test get_mesh_entity_tag(g, Cells(), :group2, :tag1) == [1, 2, 3, 7] # Sorted.
        @test mesh_entity_has_tag(g, Cells(), :group, :tag1, 1) == true
        @test mesh_entity_has_tag(g, Cells(), :group, :tag1, 4) == false
        @test mesh_entity_has_tag(g, Cells(), :group, :tag1, 3) == true
        @test_throws "Tag group2.tag3 not found in Cells()." get_mesh_entity_tag(g, Cells(), :group2, :tag3)
        @test ismissing(get_mesh_entity_tag(g, Cells(), :group2, :tag3, throw = false))
    end
end

import Jutul: check_equal_perm
@testset "check_equal_perm" begin
    @test check_equal_perm(SVector(1, 2, 3), SVector(1, 2, 3))
    @test check_equal_perm(SVector(1, 2, 3), SVector(2, 3, 1))
    @test check_equal_perm(SVector(1, 2, 3), SVector(3, 1, 2))
    @test !check_equal_perm(SVector(1, 2, 3), SVector(3, 2, 1))
    @test !check_equal_perm(SVector(1, 2, 3), SVector(3, 2, 1))
    @test !check_equal_perm(SVector(1, 2, 3), SVector(2, 1, 3))
    @test !check_equal_perm(SVector(1, 2, 3), SVector(1, 3, 2))
    @test !check_equal_perm(SVector(1, 2, 3), SVector(1, 2, 5))
    @test check_equal_perm(SVector(1, 2, 3, 4), SVector(1, 2, 3, 4))
    @test check_equal_perm(SVector(1, 2, 3, 4), SVector(2, 3, 4, 1))
    @test check_equal_perm(SVector(1, 2, 3, 4), SVector(3, 4, 1, 2))
    @test check_equal_perm(SVector(1, 2, 3, 4), SVector(4, 1, 2, 3))
end

import Jutul: BlendingParameter, BlendingVariable
@testset "BlendingVariable" begin
    mock_state = (
        A = repeat([1.0, 2, 3], 1, 5),
        B = repeat([10.0, 20, 30], 1, 5),
        BlendingParameter = range(1.0, 2.0, length = 5)
    )

    V = similar(mock_state.A)
    bvar = BlendingVariable([:A, :B], 3)
    V = Jutul.update_secondary_variable!(V, bvar, nothing, mock_state)
    @test V[:, 1] ≈ [1, 2, 3] atol = 1e-2
    @test V[:, end] ≈ [10, 20, 30] atol = 1e-2
    @test all(V[:, 3] .< [10, 20, 30])
    @test all(V[:, 3] .> [1, 2, 3])
end

import Jutul.AdjointsDI: devectorize_nested, vectorize_nested, devectorize_nested!
@testset "devectorize_nested" begin
    tmpcell = Dict(
        "SomeMaterial" => Dict(
            "Density" => 10.0,
            "DiffusionCoefficient" => 20.0,
            "Porosity" => 30.0,
            "Conductivity" => 40.0,
            "SomeString" => "Hello, world"
        ),
        "Electrolyte" => Dict(
            "Density" => 50.0,
            "DiffusionCoefficient" => 60.0,
            "Porosity" => [70.0, 80.0, 90.0],
            "Conductivity" => 100.0,
        ),
        "Separator" => Dict(
            "Density" => 110.0,
            "DiffusionCoefficient" => 120.0,
            "Porosity" => 130.0,
            "Conductivity" => 140.0,
        ),
        "SomeFloat" => 150.0,
        "SomeInt" => 160,
        "DeeplyNested" => Dict(
            "SomeMaterial" => Dict(
                "Density" => 170.0,
                "DiffusionCoefficient" => 180.0,
                "Porosity" => 190.0,
                "Conductivity" => 200.0,
            ),
            "Electrolyte" => Dict(
                "Density" => 210.0,
                "DiffusionCoefficient" => 220.0,
                "Porosity" => 230.0,
                "Conductivity" => 240.0,
            ),
        ),
    )

    x, s = vectorize_nested(tmpcell)
    @test minimum(diff(s.offsets)) == 1
    @test length(x) == 23

    d = devectorize_nested(x, s)
    d2 = devectorize_nested!(deepcopy(tmpcell), x, s)
    for di in [d, d2]
        @test di["DeeplyNested"]["SomeMaterial"]["Density"] == tmpcell["DeeplyNested"]["SomeMaterial"]["Density"]
        @test di["Electrolyte"]["Porosity"] == tmpcell["Electrolyte"]["Porosity"]
    end

    x, s = vectorize_nested(tmpcell, active = [["SomeFloat"]])
    @test length(x) == 1
    @test x == [150.0]
    x, s = vectorize_nested(tmpcell, active = [["DeeplyNested", "SomeMaterial"]])
    @test length(x) == 4
    @test sort(x) == [170.0, 180.0, 190.0, 200.0]
    x, s = vectorize_nested(tmpcell, active = [["DeeplyNested", "SomeMaterial", "Density"], ["DeeplyNested", "SomeMaterial", "Conductivity"]])
    @test length(x) == 2
    @test sort(x) == [170.0, 200.0]
end
