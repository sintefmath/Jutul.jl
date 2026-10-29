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
    @test csr * [1.0, 2.0, 3.0] == matrix * [1.0, 2.0, 3.0]
end

@testset "Parallel ILU CSR storage type" begin
    n = 6
    diagonal = @SMatrix [4.0 0.2; 0.1 3.0]
    off_diagonal = @SMatrix [-1.0 0.0; 0.0 -1.0]
    rows = vcat(1:n, 1:(n - 1), 2:n)
    columns = vcat(1:n, 2:n, 1:(n - 1))
    values = vcat(fill(diagonal, n), fill(off_diagonal, 2 * n - 2))
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
            test_mat[j, i] = i + (j - 1) * n
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

@testset "Compact block Jacobian alignment" begin
    for is_adjoint in (false, true), np in (1, 2, 3), ne in unique((1, np))
        layout = Jutul.BlockMajorLayout(is_adjoint)
        context = DefaultContext(matrix_layout = layout)
        cache = CompactAutoDiffCache(ne, 2, np; context)
        pos = cache.jacobian_positions
        @test pos isa Jutul.BlockJacobianPositions
        @test eltype(pos.first_positions) == Int
        @test size(pos.first_positions) == (1, 2)
        @test sizeof(pos.first_positions) * ne * np == sizeof(Matrix(pos))

        # Include outer system offsets and an equation group inside a larger block.
        # Explicit zeros must be retained as part of the sparse pattern.
        jac = sparse(
            repeat(1:4, 4), repeat(1:4, inner = 4),
            fill(zero(SMatrix{np, np, Float64}), 16), 4, 4
        )
        target, source = [1, 2], [2, 1]
        equation_offset = ne < np ? 2 * (np - ne) : 0
        Jutul.injective_alignment!(
            cache, nothing, jac, Cells(), context;
            target_index = target, source_index = source,
            row_offset = 1, column_offset = 1, target_offset = equation_offset
        )
        for i in 1:2, e in 1:ne, d in 1:np
            expected = Jutul.find_jac_position(
                jac, target[i], source[i],
                1, 1, equation_offset, 0, e, d, 2, 2, ne, np, context
            )
            @test Jutul.get_jacobian_pos(cache, i, e, d) == expected
        end

        generic = Jutul.GenericAutoDiffCache(
            eltype(cache.entries), ne,
            Cells(), [[1, 2], [2]], 2, 2; context
        )
        @test generic.jacobian_positions isa Jutul.BlockJacobianPositions
        extra = Jutul.create_extra_alignment((Cells = generic,))
        @test extra.Cells isa Jutul.BlockJacobianPositions
        full = Jutul.create_extra_alignment((Cells = generic,); matching_layouts = false)
        @test full.Cells isa Matrix{Int}
        @test size(full.Cells) == size(generic.jacobian_positions)

        # Inactive connections must return zero for every derivative.
        Jutul.set_jacobian_pos!(pos, 1, 1, 1, np, 0)
        @test all(iszero, pos[:, 1])
        @test_throws BoundsError pos[ne * np + 1, 1]
    end

    for layout in (Jutul.EquationMajorLayout(), Jutul.EntityMajorLayout())
        cache = CompactAutoDiffCache(
            2, 3, 2;
            context = DefaultContext(matrix_layout = layout)
        )
        @test cache.jacobian_positions isa Matrix{Int}
        @test size(cache.jacobian_positions) == (4, 3)
    end
    @test Jutul.allocate_jacobian_positions(
        Int, 3, 2, 4,
        Jutul.BlockMajorLayout()
    ) isa Matrix{Int}
end
