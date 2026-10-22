using Jutul, Test, LinearAlgebra, SparseArrays
using Jutul.KAPreconditioners
using JLArrays

function reference_hybrid!(x, A, b, owners; forward = true, l1 = true, damping = 1.0, steps = 1)
    for _ in 1:steps
        old = copy(x)
        for p in 1:maximum(owners; init = 1)
            rows = findall(==(p), owners)
            forward || reverse!(rows)
            for i in rows
                diagonal = A[i, i]
                remote = sum(abs(A[i, j]) for j in eachindex(x) if owners[j] != p; init = zero(eltype(x)))
                scale = copysign(abs(diagonal) + (l1 ? remote / 2 : 0), diagonal)
                value = b[i]
                for j in eachindex(x)
                    value -= A[i, j] * (owners[j] == p ? x[j] : old[j])
                end
                x[i] += damping * value / scale
            end
        end
    end
    return x
end

function hybrid_test_poisson(n)
    line = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
    edge = spdiagm(-1 => fill(-1.0, n - 1), 1 => fill(-1.0, n - 1))
    unit = sparse(1.0I, n, n)
    return kron(unit, line) + kron(edge, unit)
end

@testset "Hybrid Gauss-Seidel reference sweeps" begin
    # Nonsymmetric, non-contiguous ownership and a negative diagonal exercise
    # both traversal directions and distinguish lagged halos from colored GS.
    dense = [
        4.0 -1 0 -0.2 0 0; -2 5 -1 0 0 0; 0 -1 4 -1 0 0;
        0 0 -2 5 -1 0; 0 0.1 0 -1 4 -1; 0 0 0 0 1 -3
    ]
    for T in (Float32, Float64), I in (Int32, Int64),
            owners in ([1, 1, 1, 2, 2, 2], [1, 2, 1, 3, 2, 3], ones(Int, 6)),
            l1 in (false, true), damping in (1.0, 0.7), steps in (1, 3), forward in (false, true)
        A = sparse(T.(dense))
        csr = csr_matrix(A; index_type = I)
        b, initial = T.(1:6), T.((1:6) ./ 10)
        config = HybridGaussSeidel(steps, damping; l1)
        state = setup_smoother(csr, config; partition = owners)
        expected = reference_hybrid!(copy(initial), A, b, owners; l1, damping, steps, forward)
        smooth_direction! = forward ? KAPreconditioners.smooth_level! : KAPreconditioners.smooth_result!
        x = copy(initial)
        smooth_direction!(x, csr, b, state, steps)
        @test x ≈ expected
        # A supplied residual must produce the same first local correction.
        x = copy(initial)
        smooth_direction!(x, csr, b, state, steps; residual = b - A * x)
        @test x ≈ expected
        # The optimized zero-initial path cannot read stale x or correction.
        fill!(x, T(NaN))
        fill!(state.correction, T(NaN))
        smooth_direction!(x, csr, b, state, steps; zero_initial = true)
        @test x ≈ reference_hybrid!(zeros(T, 6), A, b, owners; l1, damping, steps, forward)
        # Repeat from identical input to catch accidentally shared halo reads.
        for _ in 1:3
            copyto!(x, initial)
            smooth_direction!(x, csr, b, state, steps)
            @test x ≈ expected
        end
    end
end

@testset "Hybrid Gauss-Seidel standalone and reuse" begin
    A = hybrid_test_poisson(5)
    owners = repeat(1:5, inner = 5)
    b = collect(1.0:25)
    config = HybridGaussSeidel(2, 0.8)
    csr = csr_matrix(A)
    state = setup_smoother(csr, config; partition = owners)
    expected = zeros(25)
    reference_hybrid!(expected, A, b, owners; steps = 2, damping = 0.8)
    reference_hybrid!(expected, A, b, owners; steps = 2, damping = 0.8, forward = false)
    @test state * b ≈ expected
    @test state \ b ≈ expected
    x = fill(NaN, 25)
    ldiv!(x, state, b)
    @test x ≈ expected
    @test size(state) == size(A)
    @test eltype(state) == Float64
    alias = copy(b)
    ldiv!(alias, state, alias)
    @test alias ≈ expected
    # The complete forward/backward operator is symmetric positive definite.
    basis = Matrix{Float64}(I, 25, 25)
    operator = hcat((state * basis[:, i] for i in 1:25)...)
    @test operator ≈ operator'
    @test isposdef(Symmetric(operator))

    schedule, halo, correction = state.schedule, state.halo_rhs, state.correction
    changed = csr_matrix(1.7A)
    @test setup_smoother(changed, config; reuse = state) === state
    @test state.schedule === schedule
    @test state.halo_rhs === halo
    @test state.correction === correction
    fresh = setup_smoother(changed, config; partition = owners)
    @test state * b ≈ fresh * b
    x = zeros(25)
    smooth!(x, state, changed, b; steps = 3)
    @test x ≈ reference_hybrid!(zeros(25), 1.7A, b, owners; steps = 3, damping = 0.8)
    alias = copy(b)
    smooth!(alias, state, changed, alias; steps = 3)
    @test alias ≈ reference_hybrid!(copy(b), 1.7A, b, owners; steps = 3, damping = 0.8)
    different_indices = csr_matrix(1.7A; index_type = Int64)
    @test_throws ArgumentError update_smoother!(state, different_indices)
    @test setup_smoother(different_indices, config; reuse = state) !== state

    different = copy(A)
    different[1, end] = 0.1
    @test_throws ArgumentError update_smoother!(state, csr_matrix(different))
    rebuilt = setup_smoother(csr_matrix(different), config; reuse = state)
    @test rebuilt !== state
    @test all(isfinite, rebuilt * b)
    # Empty levels and small serial levels need no halo storage.
    small = setup_smoother(csr, HybridGaussSeidel())
    @test length(small.schedule.offsets) == 2
    @test isempty(small.halo_rhs)
    @test small.schedule.rows isa UnitRange
    empty_state = setup_smoother(spzeros(0, 0), HybridGaussSeidel())
    @test isempty(empty_state * Float64[])
end

@testset "Hybrid Gauss-Seidel duplicate diagonals" begin
    for I in (Int32, UInt32)
        A = Jutul.StaticSparsityMatrixCSR(
            3, 3, I[1, 4, 6, 7],
            I[1, 1, 2, 1, 2, 3], [1.5, 2.5, -1.0, -1.0, 3.0, 1.0]
        )
        state = setup_smoother(A, HybridGaussSeidel(); partition = [1, 2, 3])
        @test state.inverse_diagonal ≈ [1 / 4.5, 1 / 3.5, 1.0]
        x = [0.2, 0.3, 0.4]
        smooth!(x, state, A, ones(3))
        @test x ≈ reference_hybrid!([0.2, 0.3, 0.4], [4.0 -1 0; -1 3 0; 0 0 1], ones(3), [1, 2, 3])
    end
end

@testset "Scaled l1 stabilizes hybrid GS" begin
    # SPD, but point Jacobi (one row per partition) diverges without scaling.
    A = fill(0.8, 8, 8) + 0.2I
    owners = collect(1:8)
    @test isposdef(A)
    stable = setup_smoother(sparse(A), HybridGaussSeidel(); partition = owners)
    plain = setup_smoother(sparse(A), HybridGaussSeidel(; l1 = false); partition = owners)
    b = ones(8)
    x, y = zeros(8), zeros(8)
    smooth!(x, stable, stable.matrix, b; steps = 40)
    smooth!(y, plain, plain.matrix, b; steps = 10)
    @test norm(b - A * x) < 1.0e-4 * norm(b)
    @test norm(b - A * y) > 100 * norm(b)
end

@testset "Hybrid Gauss-Seidel AMG and wrappers" begin
    A = hybrid_test_poisson(24)
    b = ones(size(A, 1))
    for partitioner in (:metis, :contiguous)
        config = HybridGaussSeidel(; npartitions = 10, min_partition_size = 16, partitioner)
        csr = csr_matrix(A)
        H = setup_amg(csr, AMGOptions(; smoother = config, coarse_size = 10))
        @test length(H.levels[1].smoother.schedule.offsets) == 11
        x = zeros(length(b))
        solve!(x, H, b; maxiter = 60, rtol = 1.0e-8)
        @test norm(b - A * x) < 1.0e-8 * norm(b)
        resetup_amg!(H, csr_matrix(1.5A), :operators)
        fill!(x, NaN)
        KAPreconditioners.apply!(x, H, b)
        @test all(isfinite, x)
        @test norm(b - 1.5A * x) < norm(b)
    end
    for pc in (
            KASmootherPreconditioner(:hybrid_gauss_seidel),
            KASmootherPreconditioner(HybridGaussSeidel(; min_partition_size = 16)),
            AMGPreconditioner(; smoother_type = :hybrid_gauss_seidel, coarse_size = 10),
        )
        Jutul.update_preconditioner!(pc, A, b, DefaultContext(), nothing)
        x = zeros(length(b))
        Jutul.apply!(x, pc, b)
        @test norm(b - A * x) < norm(b)
        Jutul.partial_update_preconditioner!(pc, 1.5A, b, DefaultContext(), nothing)
        Jutul.apply!(x, pc, b)
        @test norm(b - 1.5A * x) < norm(b)
    end
end

@testset "Hybrid Gauss-Seidel validation" begin
    for kwargs in ((; npartitions = 0), (; min_partition_size = 0), (; partitioner = :invalid))
        @test_throws ArgumentError HybridGaussSeidel(; kwargs...)
    end
    @test_throws ArgumentError HybridGaussSeidel(0)
    @test_throws ArgumentError HybridGaussSeidel(1, 1.1)
    A = csr_matrix(hybrid_test_poisson(2))
    config = HybridGaussSeidel()
    @test_throws DimensionMismatch setup_smoother(A, config; partition = [1])
    @test_throws ArgumentError setup_smoother(A, config; partition = [0, 0, 1, 1])
    @test_throws ArgumentError setup_smoother(A, config; partition = [1, 3, 1, 3])
    @test_throws DimensionMismatch setup_smoother(spzeros(2, 3), config)
    @test_throws ArgumentError setup_smoother(spzeros(2, 2), config)
    @test_throws ArgumentError setup_smoother(sparse([NaN 0; 0 1]), config)
    @test_throws ArgumentError setup_smoother(sparse(ComplexF64[1 0; 0 1]), config)
    @test_throws ArgumentError setup_smoother(csr_matrix(hybrid_test_poisson(2); backend = JLBackend()), config)
    state = setup_smoother(A, config)
    @test_throws DimensionMismatch state * ones(3)
    @test_throws ArgumentError smooth!(zeros(4), state, A, ones(4); steps = 0)
end
