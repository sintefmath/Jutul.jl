using Jutul, Test

@testset "SparsityTracingWrapper" begin
    n = 10
    m = 3
    test_mat = zeros(m, n)
    test_vec = zeros(n)

    # Use NaN to see what values get tagged for testing purposes
    internal_vec = collect(-1:-1:-n)

    for i in 1:n
        test_vec[i] = i
        for j in 1:m
            test_mat[j, i] = i + (j-1)*n
        end
    end

    vec_st = Jutul.SparsityTracingWrapper(test_vec, internal_vec)
    for i in 1:n
        v = vec_st[i]
        @test v == test_vec[i]*-i
    end

    mat_st = Jutul.SparsityTracingWrapper(test_mat, internal_vec)
    for i in 1:n
        for j in 1:m
            v = mat_st[j, i]
            @test v == test_mat[j, i]*-i
        end
    end

    # Test ranges for vector
    rng = 2:7
    tmp = vec_st[rng]
    @test size(tmp) == size(test_vec[rng])

    for (i, ix) in enumerate(rng)
        @test tmp[i] == vec_st[ix]
    end

    # Test subranges
    tmp2 = mat_st[:, rng]
    @test size(tmp2) == size(test_mat[:, rng])
    for (i, ix) in enumerate(rng)
        for j in 1:m
            @test tmp2[j, i] == mat_st[j, ix]
        end
    end
end
##
@testset "ad_tags" begin
    v = allocate_array_ad(1, diag_pos = 1, tag = Cells())
    @test Jutul.value(v[1]) isa Float64
    @test Jutul.value(v) isa Vector{Float64}
    v = allocate_array_ad(1, diag_pos = 1, tag = Cells())
    @test Jutul.unpack_tag(v) == Cells()
end
