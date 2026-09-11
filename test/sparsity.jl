using Jutul, Test
using SparseArrays, LinearAlgebra
using StaticArrays

@testset "StaticSparsityMatrixCSR storage" begin
    matrix = sparse([1, 1, 2, 3], [1, 3, 2, 1], [2.0, -1.0, 4.0, 3.0], 3, 3)
    csr = Jutul.StaticSparsityMatrixCSR(copy(matrix'))

    @test !hasfield(typeof(csr), :At)
    @test Matrix(csr) == Matrix(matrix)
    rows, columns, values = findnz(csr)
    @test sparse(rows, columns, values, size(csr)...) == matrix
    @test csr*[1.0, 2.0, 3.0] == matrix*[1.0, 2.0, 3.0]
end

@testset "Parallel ILU CSR storage type" begin
    n = 6
    diagonal = @SMatrix [4.0 0.2; 0.1 3.0]
    off_diagonal = @SMatrix [-1.0 0.0; 0.0 -1.0]
    rows = vcat(1:n, 1:(n-1), 2:n)
    columns = vcat(1:n, 2:n, 1:(n-1))
    values = vcat(fill(diagonal, n), fill(off_diagonal, 2*n-2))
    matrix = sparse(rows, columns, values, n, n)
    csr = Jutul.StaticSparsityMatrixCSR(
        copy(matrix');
        nthreads = 2,
        minbatch = 1,
        thread_type = :batch
    )

    factor = Jutul.ilu0_csr(csr, [1, 1, 1, 2, 2, 2])
    factor_type = typeof(first(factor.factors))
    @test all(f -> typeof(f) === factor_type, factor.factors)

    Jutul.ilu0_csr!(factor, csr)
    right_hand_side = fill(@SVector([1.0, 2.0]), n)
    solution = similar(right_hand_side)
    ldiv!(solution, factor, right_hand_side)
    @test all(x -> all(isfinite, x), solution)
end

@testset "SparsityTracingWrapper" begin
    n = 10
    m = 3
    test_mat = zeros(m, n)
    test_vec = zeros(n)

    for i in 1:n
        test_vec[i] = i
        for j in 1:m
            test_mat[j, i] = i + (j-1)*n
        end
    end

    vec_st = Jutul.SparsityTracingWrapper(test_vec)
    for i in 1:n
        v = vec_st[i]
        @test v isa Jutul.ST.ADval
        @test v.derivnode.index == i
        @test v.val == test_vec[i]
    end

    mat_st = Jutul.SparsityTracingWrapper(test_mat)
    for i in 1:n
        for j in 1:m
            v = mat_st[j, i]
            @test v isa Jutul.ST.ADval
            @test v.derivnode.index == i
            @test v.val == test_mat[j, i]
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

@testset "ad_tags" begin
    v = allocate_array_ad(1, diag_pos = 1, tag = Cells())
    @test Jutul.value(v[1]) isa Float64
    @test Jutul.value(v) isa Vector{Float64}
    v = allocate_array_ad(1, diag_pos = 1, tag = Cells())
    @test Jutul.unpack_tag(v) == Cells()
end
