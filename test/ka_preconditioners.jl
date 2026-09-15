using Jutul
using Jutul.KAPreconditioners
using JLArrays
import Jutul.Krylov: gmres
using LinearAlgebra
using SparseArrays
using Jutul.StaticArrays: SMatrix, SVector
using Test

function object_id_or_zero(value)
    if isnothing(value)
        UInt(0)
    else
        objectid(value.nzval)
    end
end

function level_memory_ids(level)
    P = if isnothing(level.P)
        nothing
    else
        (objectid(level.P.rowptr), objectid(level.P.colval), objectid(level.P.nzval))
    end
    Pt = if isnothing(level.Pt)
        nothing
    else
        (objectid(level.Pt.offsets), objectid(level.Pt.fine_rows), objectid(level.Pt.p_indices))
    end
    G = if isnothing(level.galerkin)
        nothing
    else
        (objectid(level.galerkin.offsets), objectid(level.galerkin.p_left),
         objectid(level.galerkin.a_index), objectid(level.galerkin.p_right))
    end
    symbolic = if isnothing(level.cf)
        nothing
    else
        (objectid(level.cf), objectid(level.coarse_map), objectid(level.strength))
    end
    (A = (objectid(level.A.rowptr), objectid(level.A.colval), objectid(level.A.nzval)),
     P, Pt, G,
     work = (objectid(level.smoother.diagonal), objectid(level.smoother.temporary),
             objectid(level.residual), objectid(level.correction), objectid(level.rhs)),
     symbolic)
end

function poisson_2d(n, scale=1.0)
    T = spdiagm(-1 => fill(-scale, n-1), 0 => fill(4scale, n), 1 => fill(-scale, n-1))
    E = spdiagm(-1 => fill(-scale, n-1), 1 => fill(-scale, n-1))
    kron(sparse(I, n, n), T) + kron(E, sparse(I, n, n))
end

function sparse_prolongation(P)
    rowptr = Array(P.rowptr)
    nnz = Int(rowptr[P.nrow+1]) - 1
    rows = Int[]
    for i in 1:P.nrow, _ in rowptr[i]:(rowptr[i+1]-1)
        push!(rows, i)
    end
    sparse(rows, Array(P.colval)[1:nnz], Array(P.nzval)[1:nnz], P.nrow, P.ncol)
end

function test_galerkin(H)
    for l in 1:(length(H.levels)-1)
        fine = KAPreconditioners.sparse_matrix(H.levels[l].A)
        coarse = KAPreconditioners.sparse_matrix(H.levels[l+1].A)
        P = sparse_prolongation(H.levels[l].P)
        @test Matrix(coarse) ≈ Matrix(P' * fine * P) rtol=1e-11 atol=1e-12
    end
end

@testset "csr_matrix" begin
    A = poisson_2d(5)
    C = csr_matrix(A)
    @test C isa Jutul.StaticSparsityMatrixCSR
    @test size(C) == size(A)
    @test Matrix(KAPreconditioners.sparse_matrix(C)) ≈ Matrix(A)
    @test C !== csr_matrix(A)
    @test eltype(C.rowptr) == Int32
    @test eltype(csr_matrix(A; index_type=Int64).rowptr) == Int64
    @test KAPreconditioners.matrix_block_size(C) == 128
    @test KAPreconditioners.matrix_block_size(
        csr_matrix(A; block_size = 32)) == 32
    x = collect(1.0:size(A, 1))
    y = similar(x)
    mul!(y, C, x)
    @test y ≈ A*x

    # Direct conversion must retain rectangular dimensions, empty rows, complex
    # values, and sorted column indices for either supported index width.
    R = sparse([1, 1, 3, 5], [4, 1, 2, 3],
               ComplexF64[2+im, -1, 3-im, 4], 5, 4)
    for index_type in (Int32, Int64)
        CR = csr_matrix(R; index_type=index_type)
        @test size(CR) == size(R)
        @test isapprox(Matrix(KAPreconditioners.sparse_matrix(CR)), Matrix(R))
        @test all(i -> issorted(CR.colval[nzrange(CR, i)]), 1:size(R, 1))
    end
end

@testset "Jutul preconditioner wrappers" begin
    A = poisson_2d(6)
    b = ones(size(A, 1))
    context = DefaultContext()

    @test AMGPreconditioner().options.smoother isa ILU0
    @test AMGPreconditioner().options.interpolation isa ExtendedIInterpolation
    @test AMGPreconditioner(:aggregation).options.interpolation isa ConstantInterpolation
    @test AMGPreconditioner(:ruge_stuben).options.interpolation isa ClassicalInterpolation

    for preconditioner in (
            AMGPreconditioner(:ruge_stuben; coarse_size = 10),
            KASmootherPreconditioner(:spai0),
            KASmootherPreconditioner(:gauss_seidel),
            KASmootherPreconditioner(:ilu0),
            KASmootherPreconditioner(:dilu))
        Jutul.update_preconditioner!(preconditioner, A, b, context, nothing)
        x = zeros(size(A, 1))
        Jutul.apply!(x, preconditioner, b)
        @test norm(b - A*x) < norm(b)

        B = copy(A)
        nonzeros(B) .*= 1.1
        Jutul.partial_update_preconditioner!(
            preconditioner, B, b, context, nothing)
        fill!(x, 0.0)
        Jutul.apply!(x, preconditioner, b)
        @test norm(b - B*x) < norm(b)
    end
end

@testset "AMG wrapper reuse modes refresh smoothers" begin
    A = poisson_2d(12)
    b = ones(size(A, 1))
    context = DefaultContext()
    preconditioner = AMGPreconditioner(:ruge_stuben;
        coarse_size = 10,
        reuse = :none,
        reuse_partial = :operators)
    Jutul.update_preconditioner!(preconditioner, A, b, context, nothing)

    hierarchy = preconditioner.factor
    old_levels = copy(hierarchy.levels)
    old_smoother_values = [copy(level.smoother.factors) for level in old_levels]
    B = copy(A)
    nonzeros(B) .*= 1.1
    Jutul.partial_update_preconditioner!(
        preconditioner, B, b, context, nothing)

    @test preconditioner.factor === hierarchy
    @test all(hierarchy.levels[i].smoother === old_levels[i].smoother
              for i in eachindex(old_levels))
    @test all(old_smoother_values[i] != hierarchy.levels[i].smoother.factors
              for i in eachindex(old_levels))

    partial_levels = copy(hierarchy.levels)
    Jutul.update_preconditioner!(preconditioner, A, b, context, nothing)
    @test preconditioner.factor === hierarchy
    @test all(hierarchy.levels[i].smoother !== partial_levels[i].smoother
              for i in eachindex(partial_levels))
end

@testset "KernelAbstractions CPU backend compatibility" begin
    # CPU(static=true) controls kernel scheduling, but it uses the same Array
    # storage as the default CPU(static=false) backend inferred from `zeros`.
    # Smoother and AMG reuse must therefore accept matrices and vectors from
    # either CPU variant.
    A = poisson_2d(4)
    ordinary = csr_matrix(A)
    static_backend = KernelAbstractions.CPU(; static = true)
    static_matrix = csr_matrix(
        copy(ordinary.rowptr), copy(ordinary.colval), copy(ordinary.nzval),
        size(ordinary, 1), size(ordinary, 2); backend = static_backend)
    b = ones(size(A, 1))

    for config in (SPAI0(), GaussSeidel(), ILU0(), DILU())
        smoother = setup_smoother(static_matrix, config)
        @test update_smoother!(smoother, ordinary) === smoother
        x = zeros(size(A, 1))
        KAPreconditioners.apply!(x, smoother, b)
        @test all(isfinite, x)
    end

    hierarchy = setup_amg(static_matrix, AMGOptions(coarse_size = 4))
    @test resetup_amg!(hierarchy, ordinary) === hierarchy
    x = zeros(size(A, 1))
    KAPreconditioners.apply!(x, hierarchy, b)
    @test norm(b - A*x) < norm(b)

    cpu_minbatch = 7
    threshold_matrix = csr_matrix(
        copy(ordinary.rowptr), copy(ordinary.colval), copy(ordinary.nzval),
        size(ordinary, 1), size(ordinary, 2);
        backend = KernelAbstractions.CPU(), block_size = cpu_minbatch)
    threshold_hierarchy = setup_amg(
        threshold_matrix, AMGOptions(coarse_size = 4))
    @test threshold_hierarchy.block_size == cpu_minbatch
    @test all(minbatch(level.A) == cpu_minbatch
        for level in threshold_hierarchy.levels)
    @test all(level.smoother.block_size == cpu_minbatch
        for level in threshold_hierarchy.levels)
end

@testset "Extended+i truncation preserves row sums" begin
    A = sparse(
        [1, 1, 1, 1, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 2, 3, 4, 5],
        [20.0, -1.0, -2.0, -3.0, -4.0, 1.0, 1.0, 1.0, 1.0],
        5, 5)
    C = csr_matrix(A)
    cf = Int8[-1, 1, 1, 1, 1]
    cmap = Int32[0, 1, 2, 3, 4]
    strong = falses(length(C.nzval))
    for k in nzrange(C, 1)
        strong[k] = C.colval[k] != 1
    end
    full = KAPreconditioners.build_prolongation(
        C, cf, cmap, 4, strong, ExtendedIInterpolation(0.0, 4, 2, false))
    truncated = KAPreconditioners.build_prolongation(
        C, cf, cmap, 4, strong, ExtendedIInterpolation(0.0, 2, 2, true))
    full_row = full.nzval[full.rowptr[1]:(full.rowptr[2]-1)]
    truncated_row = truncated.nzval[
        truncated.rowptr[1]:(truncated.rowptr[2]-1)]
    @test length(truncated_row) == 2
    @test sum(full_row) ≈ 0.5
    @test sum(truncated_row) ≈ sum(full_row)
end

@testset "Jutul simulation with KA AMG" begin
    grid = CartesianMesh((3, 3), (1.0, 1.0))
    model = SimulationModel(DiscretizedDomain(grid), SimpleHeatSystem())
    initial_state = setup_state(model, Dict(:T => collect(range(0.0, 1.0; length = 9))))
    simulator = Simulator(model; state0 = initial_state)
    linear_solver = GenericKrylov(:bicgstab;
        preconditioner = AMGPreconditioner(:aggregation;
            coarse_size = 4))
    states, = simulate(simulator, [1.0]; linear_solver, info_level = -1)
    @test length(states) == 1
end

@testset "coarsening and pure cycles" begin
    A = poisson_2d(14)
    b = ones(size(A, 1))
    configurations = (
        AMGOptions(coarsening=Aggregation(0.25), coarse_size=12),
        AMGOptions(coarsening=RugeStuben(0.25), coarse_size=12),
        AMGOptions(coarsening=HMIS(0.5),
                   interpolation=ExtendedIInterpolation(0.0, 4, 2, false),
                   coarse_size=12),
    )
    @test configurations[1].interpolation isa ConstantInterpolation
    @test configurations[2].interpolation isa ClassicalInterpolation
    @test configurations[3].interpolation isa ExtendedIInterpolation
    for options in configurations
        H = setup_amg(A, options)
        @test length(H.levels) >= 2
        @test H.levels[end].coarse_solver isa KAPreconditioners.CoarseLUState
        x = zeros(size(A, 1))
        r0 = norm(b - A*x)
        for _ in 1:4
            cycle!(x, H, b)
        end
        @test norm(b - A*x) < r0
    end
end

@testset "coarse LU" begin
    @test AMGOptions().coarse_solver == :lu
    @test AMGOptions().coarse_size == 50
    M = [0.0 2.0 1.0; 1.0 1.0 0.0; 2.0 0.0 1.0]
    rhs = [1.0, -2.0, 3.0]
    factorization = lu!(copy(M))
    solution = similar(rhs)
    ldiv!(solution, factorization, rhs)
    @test M * solution ≈ rhs rtol=1e-13 atol=1e-13

    A = poisson_2d(10)
    H = setup_amg(A, AMGOptions(coarse_size=10))
    b = ones(size(A, 1))
    x = zeros(size(A, 1))
    KAPreconditioners.apply!(x, H, b)
    fine = H.levels[end-1]
    coarse = H.levels[end]
    @test Array(coarse.A * fine.correction) ≈ Array(fine.rhs) rtol=1e-12 atol=1e-12

    Hs = setup_amg(A, AMGOptions(coarse_size=10, coarse_solver=:spai0))
    @test isnothing(Hs.levels[end].coarse_solver)
    @test_throws ArgumentError setup_amg(A, AMGOptions(coarse_solver=:invalid))
end

@testset "asymmetric Galerkin product" begin
    A = poisson_2d(8)
    A[1, 2] = -0.2
    A[2, 1] = -1.8
    A[10, 11] = -0.35
    A[11, 10] = -1.65
    H = setup_amg(A, AMGOptions(coarse_size=8, max_row_sum=0.9))
    test_galerkin(H)
end

@testset "in-place resetup" begin
    A = poisson_2d(12)
    H = setup_amg(A, AMGOptions(coarsening=HMIS(0.5), coarse_size=10))
    array_ids = [(objectid(level.A.nzval), object_id_or_zero(level.P)) for level in H.levels]
    # Make preservation observable even if the replacement matrix is a scaled
    # version of the original one.
    H.levels[1].P.nzval[2] *= 0.9
    interpolation_values = map(H.levels) do level
        if isnothing(level.P)
            return nothing
        else
            return copy(level.P.nzval)
        end
    end
    B = copy(A)
    nonzeros(B) .*= 1.7
    resetup_amg!(H, B, :operators)
    @test array_ids == [(objectid(level.A.nzval), object_id_or_zero(level.P)) for level in H.levels]
    current_interpolation_values = map(H.levels) do level
        if isnothing(level.P)
            return nothing
        else
            return level.P.nzval
        end
    end
    @test interpolation_values == current_interpolation_values
    @test Array(H.levels[1].A.nzval) ≈ nonzeros(csr_matrix(B))
    test_galerkin(H)
    xb = zeros(size(B, 1))
    KAPreconditioners.apply!(xb, H, ones(size(B, 1)))
    @test Array(H.levels[end].A * H.levels[end-1].correction) ≈
          Array(H.levels[end-1].rhs) rtol=1e-12 atol=1e-12
    resetup_amg!(H, A, :sparsity)
    @test array_ids == [(objectid(level.A.nzval), object_id_or_zero(level.P)) for level in H.levels]
    @test H.levels[1].P.nzval != interpolation_values[1]
    test_galerkin(H)
    @test_throws ArgumentError resetup_amg!(H, spdiagm(0 => ones(size(A, 1))), :operators)

    memory_ids = [level_memory_ids(level) for level in H.levels]
    finest_id = objectid(H.levels[1].A.nzval)
    resetup_amg!(H, B, :memory)
    @test objectid(H.levels[1].A.nzval) == finest_id
    @test memory_ids == [level_memory_ids(level) for level in H.levels]
    resetup_amg!(H, A, :none)
    @test objectid(H.levels[1].A.nzval) != finest_id
end

@testset "symbolic memory reset and poisoned work buffers" begin
    A = poisson_2d(12)
    options = AMGOptions(coarsening=HMIS(0.5), coarse_size=10)
    H = setup_amg(A, options)
    B = copy(A)
    rows = rowvals(B)
    @inbounds for j in axes(B, 2), p in nzrange(B, j)
        i = rows[p]
        factor = if i == j
            1.3
        elseif isodd(i + j)
            0.4
        else
            1.6
        end
        B.nzval[p] *= factor
    end
    resetup_amg!(H, B, :memory)
    @test isapprox(Matrix(KAPreconditioners.sparse_matrix(H.levels[1].A)), Matrix(B))
    test_galerkin(H)

    b = ones(size(B, 1))
    reference = zeros(size(B, 1))
    KAPreconditioners.apply!(reference, setup_amg(B, options), b)
    x = fill(NaN, size(B, 1))
    for level in H.levels
        fill!(level.smoother.temporary, NaN)
        fill!(level.residual, NaN)
        fill!(level.correction, NaN)
        fill!(level.rhs, NaN)
    end
    KAPreconditioners.apply!(x, H, b)
    @test all(isfinite, x)
    @test isapprox(x, reference; rtol=1e-12, atol=1e-12)

    # Level 1 is fixed by the discretization even though all coarse symbolic
    # data is rebuilt in :memory mode.
    level1_ids = (objectid(H.levels[1].A.rowptr),
                  objectid(H.levels[1].A.colval), objectid(H.levels[1].A.nzval))
    resetup_amg!(H, A, :memory)
    @test level1_ids == (objectid(H.levels[1].A.rowptr),
                         objectid(H.levels[1].A.colval), objectid(H.levels[1].A.nzval))
    @test isapprox(Matrix(KAPreconditioners.sparse_matrix(H.levels[1].A)), Matrix(A))
    test_galerkin(H)

    changed_graph = copy(A)
    changed_graph[1, end] = -0.05
    @test_throws ArgumentError resetup_amg!(H, changed_graph, :memory)
end

@testset "standalone and Krylov solves" begin
    A = poisson_2d(18)
    b = ones(size(A, 1))
    H = setup_amg(A, AMGOptions(coarse_size=12))
    x = zeros(size(A, 1))
    x, iterations = solve!(x, H, b; rtol=1e-7, maxiter=60)
    @test iterations <= 60
    @test norm(b-A*x)/norm(b) < 1e-6

    Hk = setup_amg(A, AMGOptions(coarse_size=12))
    # Krylov's left-preconditioned stopping norm is not the true residual, so
    # request a tighter internal tolerance before checking the latter.
    xk, stats = gmres(A, b; M=Hk, ldiv=true, rtol=1e-10, itmax=100)
    @test norm(b-A*xk)/norm(b) < 1e-7
    @test stats.niter < 100
end

@testset "input validation" begin
    A = poisson_2d(4)
    H = setup_amg(A)
    @test_throws ArgumentError resetup_amg!(H, A, :invalid)
    @test_throws DimensionMismatch setup_amg(sparse(ones(3, 2)))
    @test_throws ArgumentError setup_amg(A, AMGOptions(max_row_sum=-0.1))
    @test_throws ArgumentError setup_amg(A, AMGOptions(
        coarsening=Aggregation(), interpolation=ClassicalInterpolation()))
    @test_throws ArgumentError setup_amg(A, AMGOptions(
        coarsening=RugeStuben(), interpolation=ConstantInterpolation()))
end

@testset "max row sum" begin
    A = sparse([1, 1, 2, 2, 2, 3, 3],
               [1, 2, 1, 2, 3, 2, 3],
               [2.0, -0.1, -1.0, 2.0, -1.0, -1.0, 2.0], 3, 3)
    C = csr_matrix(A)
    original = copy(C.nzval)
    regular = Array(KAPreconditioners.strength(C, 0.25, 1.0))
    weakened = Array(KAPreconditioners.strength(C, 0.25, 0.9))
    row1 = C.rowptr[1]:(C.rowptr[2]-1)
    @test any(regular[row1])
    @test !any(weakened[row1])
    @test C.nzval == original
    @test AMGOptions(max_row_sum=0.9).max_row_sum == 0.9
end

@testset "HMIS and Extended+i semantics" begin
    F = sparse([1, 1, 1, 2, 3], [1, 2, 3, 2, 3], [2.0, 1.0, 0.5, 1.0, 1.0], 3, 3)
    CF = csr_matrix(F)
    fallback_strength = KAPreconditioners.strength(CF, 0.5, 1.0)
    row1 = CF.rowptr[1]:(CF.rowptr[2]-1)
    by_column = Dict(CF.colval[k] => fallback_strength[k] for k in row1)
    @test by_column[Int32(2)]
    @test !by_column[Int32(3)] # strict `>` at exactly theta * max

    # A one-dimensional graph has enough ties to distinguish the RS+PMIS
    # hybrid from the old descending-degree greedy approximation.
    A = spdiagm(-1 => fill(-1.0, 11), 0 => fill(2.0, 12), 1 => fill(-1.0, 11))
    C = csr_matrix(A)
    strong = KAPreconditioners.strength(C, 0.5, 1.0)
    cf1, cmap1, nc1 = KAPreconditioners.cf_split(C, strong, HMIS(0.5))
    cf2, cmap2, nc2 = KAPreconditioners.cf_split(C, strong, HMIS(0.5))
    @test cf1 == cf2
    @test cmap1 == cmap2
    @test nc1 == nc2 == count(==(1), cf1)
    @test all(eachindex(cf1)) do i
        if cf1[i] == 1
            cmap1[i] > 0
        else
            cmap1[i] == 0
        end
    end

    # Dependency weakening must produce an empty interpolation row, rather
    # than the previous arbitrary connection to coarse column one.
    W = sparse([1, 1, 2, 2], [1, 2, 1, 2], [2.0, -0.1, -1.0, 2.0], 2, 2)
    CW = csr_matrix(W)
    weak = KAPreconditioners.strength(CW, 0.25, 0.9)
    P = KAPreconditioners.build_prolongation(CW, Int8[-1, 1], Int32[0, 1], 1, weak,
                                    ExtendedIInterpolation(0.0, 4, 2, true))
    @test P.rowptr[1] == P.rowptr[2]
    @test P.rowptr[3] - P.rowptr[2] == 1
end

@testset "KernelAbstractions device hierarchy" begin
    A = poisson_2d(10)
    C = csr_matrix(A)
    backend = JLBackend()
    D = csr_matrix(A; backend=backend)
    @test D.rowptr isa JLArray
    @test Matrix(KAPreconditioners.sparse_matrix(D)) ≈ Matrix(A)
    H = setup_amg(D, AMGOptions(coarse_size=10))
    @test H.levels[1].A.nzval isa JLArray
    @test all(level -> level.A.nzval isa JLArray, H.levels)
    coarse_solver = H.levels[end].coarse_solver
    @test coarse_solver isa KAPreconditioners.HostLUState
    @test coarse_solver.factorization isa LU
    @test coarse_solver.factorization.factors isa Matrix
    b = JLArray(ones(size(A, 1)))
    x = JLArray(zeros(size(A, 1)))
    for _ in 1:4
        cycle!(x, H, b)
    end
    @test norm(ones(size(A, 1)) - A*Array(x)) < norm(ones(size(A, 1)))

    D2 = csr_matrix(copy(D.rowptr), copy(D.colval), 1.2 .* D.nzval,
                   size(D, 1), size(D, 2); backend=backend)
    ids = map(level -> objectid(level.A.nzval), H.levels)
    resetup_amg!(H, D2, :sparsity)
    @test ids == map(level -> objectid(level.A.nzval), H.levels)

    DB = csr_matrix(copy(D.rowptr), copy(D.colval),
                   D.nzval .* JLArray([if isodd(i) 0.8 else 1.3 end for i in eachindex(D.nzval)]),
                   size(D, 1), size(D, 2); backend=backend)
    level1_ids = (objectid(H.levels[1].A.rowptr), objectid(H.levels[1].A.colval),
                  objectid(H.levels[1].A.nzval))
    resetup_amg!(H, DB, :memory)
    test_galerkin(H)
    @test level1_ids == (objectid(H.levels[1].A.rowptr), objectid(H.levels[1].A.colval),
                         objectid(H.levels[1].A.nzval))
    resetup_amg!(H, D, :memory)
    test_galerkin(H)

    # Exercise every method-specific interpolation reset on device arrays.
    for coarsening in (RugeStuben(), Aggregation())
        method_options = AMGOptions(coarsening=coarsening, coarse_size=10)
        method_hierarchy = setup_amg(D, method_options)
        resetup_amg!(method_hierarchy, D2, :sparsity)
        method_x = JLArray(zeros(size(A, 1)))
        cycle!(method_x, method_hierarchy, b)
        @test all(isfinite, Array(method_x))
        test_galerkin(method_hierarchy)
    end
end

@testset "standalone ILU smoothers" begin
    A = poisson_2d(5)
    C = csr_matrix(A)
    b = ones(size(A, 1))

    for config in (ILU0(), DILU())
        state = setup_smoother(C, config)
        @test size(state) == size(A)
        @test eltype(state) == Float64

        x = zeros(size(A, 1))
        KAPreconditioners.apply!(x, state, b)
        @test norm(b - A*x) < norm(b)
        @test state \ b ≈ x
        product = similar(x)
        mul!(product, state, b)
        @test product ≈ x

        fill!(x, 0.25)
        initial_residual = norm(b - A*x)
        smooth!(x, state, C, b; steps=2)
        @test norm(b - A*x) < initial_residual

        factor_id = if state isa KAPreconditioners.ILU0State
            objectid(state.factors)
        else
            objectid(state.values)
        end
        diagonal_id = objectid(state.inverse_diagonal)
        factor_rows_id = objectid(state.factor_rows)
        upper_rows_id = objectid(state.upper_rows)

        B = copy(A)
        nonzeros(B) .*= 1.3
        D = csr_matrix(B)
        @test setup_smoother(D, config; reuse=state) === state
        @test objectid(state.inverse_diagonal) == diagonal_id
        @test objectid(state.factor_rows) == factor_rows_id
        @test objectid(state.upper_rows) == upper_rows_id
        if state isa KAPreconditioners.ILU0State
            @test objectid(state.factors) == factor_id
        else
            @test objectid(state.values) == factor_id
        end

        changed_pattern = csr_matrix(spdiagm(0 => fill(2.0, size(A, 1))))
        @test_throws ArgumentError update_smoother!(state, changed_pattern)
    end

    for config in (ILU0(), DILU())
        H = setup_amg(A, AMGOptions(smoother=config, coarse_size=8))
        x = zeros(size(A, 1))
        cycle!(x, H, b)
        @test norm(b - A*x) < norm(b)
    end
end

@testset "ILU smoothers on KernelAbstractions arrays" begin
    A = poisson_2d(5)
    backend = JLBackend()
    C = csr_matrix(A; backend=backend)
    b = JLArray(ones(size(A, 1)))

    for config in (ILU0(), DILU())
        state = setup_smoother(C, config)
        x = JLArray(zeros(size(A, 1)))
        KAPreconditioners.apply!(x, state, b)
        @test norm(ones(size(A, 1)) - A*Array(x)) < norm(ones(size(A, 1)))

        host_state = setup_smoother(csr_matrix(A), config)
        host_x = zeros(size(A, 1))
        KAPreconditioners.apply!(host_x, host_state, ones(size(A, 1)))
        @test Array(state.inverse_diagonal) ≈ host_state.inverse_diagonal
        @test Array(x) ≈ host_x

        fill!(x, 0.25)
        initial_residual = norm(ones(size(A, 1)) - A*Array(x))
        smooth!(x, state, C, b; steps=2)
        @test norm(ones(size(A, 1)) - A*Array(x)) < initial_residual
    end
end

@testset "fused ILU post-smoothing" begin
    A = poisson_2d(5)
    initial = fill(0.25, size(A, 1))
    rhs = ones(size(A, 1))
    for backend in (KernelAbstractions.CPU(), JLBackend())
        C = csr_matrix(A; backend=backend)
        if backend isa KernelAbstractions.CPU
            b = rhs
        else
            b = JLArray(rhs)
        end
        for config in (ILU0(2), DILU(2))
            state = setup_smoother(C, config)
            if backend isa KernelAbstractions.CPU
                reference = copy(initial)
            else
                reference = JLArray(initial)
            end
            fused = copy(reference)
            smooth!(reference, state, C, b; steps=2)
            KAPreconditioners.smooth_result!(fused, C, b, state, 2)
            @test Array(fused) ≈ Array(reference)
        end
    end
end

@testset "DILU recurrence and scale invariance" begin
    A = sparse([4.0 1.0 0.0; 2.0 5.0 3.0; 0.0 4.0 6.0])
    d1 = inv(A[1, 1])
    d2 = inv(A[2, 2] - A[2, 1]*d1*A[1, 2])
    d3 = inv(A[3, 3] - A[3, 2]*d2*A[2, 3])
    expected_diagonal = [d1, d2, d3]
    rhs = [1.0, -2.0, 4.0]

    y1 = d1*rhs[1]
    y2 = d2*(rhs[2] - A[2, 1]*y1)
    y3 = d3*(rhs[3] - A[3, 2]*y2)
    z3 = y3
    z2 = y2 - d2*A[2, 3]*z3
    z1 = y1 - d1*A[1, 2]*z2
    expected = [z1, z2, z3]

    for backend in (KernelAbstractions.CPU(), JLBackend())
        C = csr_matrix(A; backend=backend)
        state = setup_smoother(C, DILU())
        backend_rhs = backend isa KernelAbstractions.CPU ? rhs : JLArray(rhs)
        result = similar(backend_rhs)
        KAPreconditioners.apply!(result, state, backend_rhs)
        @test Array(state.inverse_diagonal) ≈ expected_diagonal
        @test Array(result) ≈ expected
    end

    # Multiplying A by a scalar must divide both D^-1 and the action of the
    # preconditioner by that scalar. In particular, small but nonsingular SI
    # coefficients must not be replaced by an absolute pivot threshold.
    scale = 1e-12
    state = setup_smoother(csr_matrix(A), DILU())
    scaled_state = setup_smoother(csr_matrix(scale*A), DILU())
    result = state \ rhs
    scaled_result = scaled_state \ rhs
    @test scaled_state.inverse_diagonal ≈ state.inverse_diagonal/scale
    @test scaled_result ≈ result/scale
end

@testset "static block smoothers" begin
    Block = SMatrix{2,2,Float64,4}
    BlockVector = SVector{2,Float64}
    diagonal = Block(4.0, 0.1, 0.2, 3.0)
    lower = Block(-1.0, 0.0, 0.1, -0.6)
    upper = Block(-0.8, -0.1, 0.0, -0.5)
    n = 8
    rows = vcat(collect(1:n), collect(2:n), collect(1:(n - 1)))
    columns = vcat(collect(1:n), collect(1:(n - 1)), collect(2:n))
    values = vcat(fill(diagonal, n), fill(lower, n - 1), fill(upper, n - 1))
    A = sparse(rows, columns, values, n, n)
    C = csr_matrix(A)
    b = [BlockVector(1.0 + 0.1*i, -0.5 + 0.05*i) for i in 1:n]

    for config in (SPAI0(), ILU0(), DILU())
        state = setup_smoother(C, config)
        x = fill(zero(BlockVector), n)
        KAPreconditioners.apply!(x, state, b)
        @test norm(b - A*x) < norm(b)

        fill!(x, BlockVector(0.2, -0.1))
        initial_residual = norm(b - A*x)
        smooth!(x, state, C, b; steps=2)
        @test norm(b - A*x) < initial_residual
    end

    backend = JLBackend()
    D = csr_matrix(A; backend=backend)
    device_b = JLArray(b)
    for config in (ILU0(), DILU())
        state = setup_smoother(D, config)
        x = JLArray(fill(zero(BlockVector), n))
        KAPreconditioners.apply!(x, state, device_b)
        @test norm(b - A*Array(x)) < norm(b)

        host_state = setup_smoother(C, config)
        host_x = fill(zero(BlockVector), n)
        KAPreconditioners.apply!(host_x, host_state, b)
        @test Array(state.inverse_diagonal) ≈ host_state.inverse_diagonal
        @test Array(x) ≈ host_x
    end
end


@testset "KA ILU0 matches serial StaticCSR" begin
    function compare_ilu(A, b)
        matrix = csr_matrix(A; index_type = Int)
        smoother = setup_smoother(matrix, ILU0())
        ka_result = similar(b)
        KAPreconditioners.apply!(ka_result, smoother, b)

        serial_factor = Jutul.ilu0_csr(matrix)
        serial_result = similar(b)
        ldiv!(serial_result, serial_factor, b)
        @test ka_result ≈ serial_result rtol = 1e-12 atol = 1e-12
        return serial_result
    end

    scalar_matrix = poisson_2d(5)
    compare_ilu(scalar_matrix, ones(size(scalar_matrix, 1)))

    Block = SMatrix{2, 2, Float64, 4}
    BlockVector = SVector{2, Float64}
    diagonal = Block(4.0, 0.1, 0.2, 3.0)
    off_diagonal = Block(-0.8, -0.1, 0.0, -0.5)
    n = 8
    rows = vcat(1:n, 2:n, 1:(n-1))
    columns = vcat(1:n, 1:(n-1), 2:n)
    values = vcat(fill(diagonal, n), fill(off_diagonal, 2*n-2))
    block_matrix = sparse(rows, columns, values, n, n)
    rhs = [BlockVector(1.0 + 0.1*i, -0.5 + 0.05*i) for i in 1:n]
    block_reference = compare_ilu(block_matrix, rhs)

    flat_rhs = reinterpret(Float64, rhs)
    wrapped_smoother = KASmootherPreconditioner(:ilu0)
    Jutul.update_preconditioner!(
        wrapped_smoother, csr_matrix(block_matrix; index_type = Int),
        flat_rhs, DefaultContext(), nothing)
    @test Jutul.operator_nrows(wrapped_smoother) == length(flat_rhs)
    flat_result = similar(flat_rhs)
    Jutul.apply!(flat_result, wrapped_smoother, flat_rhs)
    @test isapprox(reinterpret(BlockVector, flat_result), block_reference;
        rtol = 1e-12, atol = 1e-12)

    device_matrix = csr_matrix(block_matrix; backend = JLBackend())
    device_rhs = JLArray(flat_rhs)
    device_result = similar(device_rhs)
    device_smoother = KASmootherPreconditioner(:ilu0)
    Jutul.update_preconditioner!(device_smoother, device_matrix,
        device_rhs, KernelAbstractionsContext(JLBackend()), nothing)
    Jutul.apply!(device_result, device_smoother, device_rhs)
    @test isapprox(reinterpret(BlockVector, Array(device_result)),
        block_reference; rtol = 1e-12, atol = 1e-12)
end
