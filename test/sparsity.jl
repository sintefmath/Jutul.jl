using Jutul, Test
using SparseArrays, LinearAlgebra
using StaticArrays

# Count actual sparse searches, including searches whose result is discarded.
struct CountingAlignmentMatrix{M}
    matrix::M
    searches::Base.RefValue{Int}
end

function Jutul.find_sparse_position(A::CountingAlignmentMatrix, row, col, layout)
    A.searches[] += 1
    return Jutul.find_sparse_position(A.matrix, row, col, layout)
end

@testset "StaticSparsityMatrixCSR storage" begin
    matrix = sparse([1, 1, 2, 3], [1, 3, 2, 1], [2.0, -1.0, 4.0, 3.0], 3, 3)
    csr = Jutul.StaticSparsityMatrixCSR(copy(matrix'))

    @test !hasfield(typeof(csr), :At)
    @test Matrix(csr) == Matrix(matrix)
    rows, columns, values = findnz(csr)
    @test sparse(rows, columns, values, size(csr)...) == matrix
    @test csr * [1.0, 2.0, 3.0] == matrix * [1.0, 2.0, 3.0]
end

@testset "Block alignment searches only stored positions" begin
    for adjoint in (false, true), ne in (1, 3), compact in (false, true)
        np = 3
        layout = Jutul.BlockMajorLayout(adjoint)
        context = DefaultContext(matrix_layout = layout)
        jac = sparse(
            repeat(1:2, 2), repeat(1:2, inner = 2),
            fill(zero(SMatrix{3, 3, Float64}), 4), 2, 2
        )
        counted = CountingAlignmentMatrix(jac, Ref(0))
        cache = CompactAutoDiffCache(ne, 2, np; context)
        positions = compact ? cache.jacobian_positions : zeros(Int, ne * np, 2)
        expected_searches = compact ? 2 : 2 * ne * np

        Jutul.injective_alignment!(
            cache, nothing, counted, Cells(), context;
            pos = positions, target_index = [2, 1], source_index = [1, 2]
        )
        @test counted.searches[] == expected_searches
        for i in 1:2, e in 1:ne, d in 1:np
            @test positions[(e - 1) * np + d, i] == Jutul.find_jac_position(
                jac, 3 - i, i, 0, 0, 0, 0, e, d, 2, 2, ne, np, context
            )
        end

        # Use a full cache as well as compact storage for the flux paths.
        flux_context = compact ? context : DefaultContext(matrix_layout = Jutul.EntityMajorLayout(adjoint))
        flux = CompactAutoDiffCache(ne, 1, np; context = flux_context)
        counted.searches[] = 0
        connections = [(self = 1, other = 2, face = 1), (self = 2, other = 1, face = 1)]
        for i in 1:2
            # A single cache entry is enough to check each directed connection.
            Jutul.align_half_face_cells(
                flux, counted, [connections[i]], 1, i,
                (2, ne, np), context, Jutul.TrivialGlobalMap(), 0, 0, 0, 0
            )
            for e in 1:ne, d in 1:np
                @test Jutul.get_jacobian_pos(flux, 1, e, d) == Jutul.find_jac_position(
                    jac, 3 - i, i, 0, 0, 0, 0, e, d, 2, 2, ne, np, context
                )
            end
        end
        @test counted.searches[] == expected_searches

        inactive = Jutul.FiniteVolumeGlobalMap([1, 2], [1], [false, true])
        counted.searches[] = 0
        Jutul.align_half_face_cells(
            flux, counted, connections, 1, 1,
            (2, ne, np), context, inactive, 0, 0, 0, 0
        )
        @test counted.searches[] == 0
        @test all(iszero, flux.jacobian_positions)

        # A generic face cache has one entity and two half-face positions.
        faces = CompactAutoDiffCache(ne, 1, np; context = flux_context, n_entities_pos = 2)
        disc = (conn_pos = [1, 2, 3], conn_data = connections)
        counted.searches[] = 0
        Jutul.half_face_flux_faces_alignment!(faces, counted, context, nothing, disc)
        @test counted.searches[] == expected_searches
        for i in 1:2, e in 1:ne, d in 1:np
            @test Jutul.get_jacobian_pos(faces, i, e, d) == Jutul.find_jac_position(
                jac, i, 1, 0, 0, 0, 0, e, d, 2, 1, ne, np, context
            )
        end

        # General finite-volume assembly aligns both sides of each face.
        acc = CompactAutoDiffCache(ne, 2, np; context = flux_context)
        cells = Jutul.GenericAutoDiffCache(
            eltype(acc.entries), ne, Cells(),
            [[1, 2]], 1, 2; context = flux_context
        )
        storage = Jutul.ConservationLawFiniteVolumeStorage(
            acc, :mass, cells,
            similar(cells.jacobian_positions), Int[], [(1, 2)]
        )
        domain = DiscretizedDomain(CartesianMesh((2, 1), (2.0, 1.0)))
        model = (context = context, domain = domain)
        counted.searches[] = 0
        law = ConservationLaw(Jutul.PotentialFlow(nothing, nothing, nothing; ad = :fvm), :mass, ne)
        Jutul.align_to_jacobian!(storage, law, counted, model, Cells())
        @test counted.searches[] == 3 * expected_searches
        for (positions, target) in ((cells.jacobian_positions, 1), (storage.face_flux_extra_alignment, 2))
            for source in 1:2, e in 1:ne, d in 1:np
                @test positions[(e - 1) * np + d, source] == Jutul.find_jac_position(
                    jac, target, source, 0, 0, 0, 0, e, d, 2, 2, ne, np, context
                )
            end
        end
    end
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
