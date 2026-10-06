using Jutul
using Jutul.KAPreconditioners
using JLArrays
using KernelAbstractions
import Jutul.Krylov: gmres
using LinearAlgebra
using SparseArrays
using Jutul.StaticArrays: SMatrix, SVector
using Test

function object_id_or_zero(value)
    return if isnothing(value)
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
        (
            objectid(level.galerkin.offsets), objectid(level.galerkin.p_left),
            objectid(level.galerkin.a_index), objectid(level.galerkin.p_right),
        )
    end
    symbolic = if isnothing(level.cf)
        nothing
    else
        (objectid(level.cf), objectid(level.coarse_map), objectid(level.strength))
    end
    return (
        A = (objectid(level.A.rowptr), objectid(level.A.colval), objectid(level.A.nzval)),
        P, Pt, G,
        work = (
            objectid(level.smoother.diagonal), objectid(level.smoother.temporary),
            objectid(level.residual), objectid(level.correction), objectid(level.rhs),
        ),
        symbolic,
    )
end

function poisson_2d(n, scale = 1.0)
    T = spdiagm(-1 => fill(-scale, n - 1), 0 => fill(4scale, n), 1 => fill(-scale, n - 1))
    E = spdiagm(-1 => fill(-scale, n - 1), 1 => fill(-scale, n - 1))
    return kron(sparse(I, n, n), T) + kron(E, sparse(I, n, n))
end

@testset "SparseLU setup and same-pattern resetup" begin
    A = poisson_2d(4)
    b = collect(1.0:size(A, 1))
    for backend in (KernelAbstractions.CPU(), JLBackend())
        matrix = csr_matrix(A; backend)
        rhs = backend isa KernelAbstractions.CPU ? b : JLArray(b)
        F = backend isa KernelAbstractions.CPU ?
            KAPreconditioners.setup_sparse_lu(matrix) :
            Jutul.KernelExecution.factorize_linear_system(lu, matrix)
        @test F isa SparseLU
        x = similar(rhs)
        ldiv!(x, F, rhs)
        @test Array(x) ≈ A \ b

        changed = csr_matrix(1.5A; backend)
        @test KAPreconditioners.resetup_sparse_lu!(F, changed) === F
        ldiv!(x, F, rhs)
        @test Array(x) ≈ (1.5A) \ b

        different = copy(A)
        different[1, 3] = 0.1
        @test_throws ArgumentError KAPreconditioners.resetup_sparse_lu!(
            F, csr_matrix(different; backend)
        )

        unpivoted = KAPreconditioners.setup_sparse_lu(matrix; pivoting = false)
        @test !unpivoted.factorization.pivoting
        ldiv!(x, unpivoted, rhs)
        @test Array(x) ≈ A \ b
    end
end

@testset "SparseLU row pivoting" begin
    # Resetup changes which row is selected as the first pivot while
    # preserving the CSR pattern, including its explicit zero diagonal.
    rows = [1, 1, 2, 2]
    cols = [1, 2, 1, 2]
    A = sparse(rows, cols, [0.0, 2.0, 3.0, 4.0], 2, 2)
    B = sparse(rows, cols, [5.0, 2.0, 3.0, 4.0], 2, 2)
    C = sparse([1, 2, 2], [2, 1, 2], [2.0, 3.0, 4.0], 2, 2)
    b = [1.0, 2.0]
    for backend in (KernelAbstractions.CPU(), JLBackend())
        matrix = csr_matrix(A; backend)
        F = KAPreconditioners.setup_sparse_lu(matrix)
        @test F.factorization.pivoting
        rhs = backend isa KernelAbstractions.CPU ? b : JLArray(b)
        x = similar(rhs)
        ldiv!(x, F, rhs)
        @test Array(x) ≈ A \ b
        @test KAPreconditioners.resetup_sparse_lu!(
            F, csr_matrix(B; backend)
        ) === F
        ldiv!(x, F, rhs)
        @test Array(x) ≈ B \ b
        missing_diagonal = KAPreconditioners.setup_sparse_lu(csr_matrix(C; backend))
        ldiv!(x, missing_diagonal, rhs)
        @test Array(x) ≈ C \ b
    end
end

@testset "Float32 KA preconditioner scalars" begin
    A = poisson_2d(5, 1.0f0)
    @test eltype(A) === Float32
    @test KAPreconditioners.matrix_scalar_type(
        SVector{2, Float32}
    ) === Float32
    @test KAPreconditioners.candidate_count(
        Float32[1, 0.6, 0.4],
        ExtendedIInterpolation(0.25, 4, 2, true)
    ) == 2

    for backend in (KernelAbstractions.CPU(), JLBackend())
        C = csr_matrix(A; backend = backend)
        @test eltype(C.nzval) === Float32
        strong = KAPreconditioners.strength(C, 0.25, 0.9)
        @test eltype(strong) === Bool

        rhs = ones(Float32, size(A, 1))
        if backend isa KernelAbstractions.CPU
            b = rhs
        else
            b = JLArray(rhs)
        end
        for config in (SPAI0(1, 0.7), ILU0(1, 0.7), DILU(1, 0.7))
            state = setup_smoother(C, config)
            @test eltype(state) === Float32
            @test KAPreconditioners.smoother_damping(state) === Float32(0.7)
            x = similar(b)
            KAPreconditioners.apply!(x, state, b)
            @test all(isfinite, Array(x))
        end

        H = setup_amg(
            C, AMGOptions(
                smoother = SPAI0(1, 0.7), coarse_size = 8
            )
        )
        @test eltype(H) === Float32
        @test all(level -> eltype(level.A.nzval) === Float32, H.levels)
        @test KAPreconditioners.smoother_damping(
            H.levels[1].smoother
        ) === Float32(0.7)
    end
end

function sparse_prolongation(P)
    rowptr = Array(P.rowptr)
    nnz = Int(rowptr[P.nrow + 1]) - 1
    rows = Int[]
    for i in 1:P.nrow, _ in rowptr[i]:(rowptr[i + 1] - 1)
        push!(rows, i)
    end
    return sparse(rows, Array(P.colval)[1:nnz], Array(P.nzval)[1:nnz], P.nrow, P.ncol)
end

function test_galerkin(H)
    for l in 1:(length(H.levels) - 1)
        fine = KAPreconditioners.sparse_matrix(H.levels[l].A)
        coarse = KAPreconditioners.sparse_matrix(H.levels[l + 1].A)
        P = sparse_prolongation(H.levels[l].P)
        @test Matrix(coarse) ≈ Matrix(P' * fine * P) rtol = 1.0e-11 atol = 1.0e-12
    end
    return
end

@testset "csr_matrix" begin
    A = poisson_2d(5)
    C = csr_matrix(A)
    @test C isa Jutul.StaticSparsityMatrixCSR
    @test size(C) == size(A)
    @test Matrix(KAPreconditioners.sparse_matrix(C)) ≈ Matrix(A)
    @test C !== csr_matrix(A)
    @test eltype(C.rowptr) == Int32
    @test eltype(csr_matrix(A; index_type = Int64).rowptr) == Int64
    @test KAPreconditioners.matrix_batch_size(C) == 128
    @test KAPreconditioners.matrix_batch_size(
        csr_matrix(A; block_size = 32)
    ) == 32
    x = collect(1.0:size(A, 1))
    y = similar(x)
    mul!(y, C, x)
    @test y ≈ A * x

    # Direct conversion must retain rectangular dimensions, empty rows, complex
    # values, and sorted column indices for either supported index width.
    R = sparse(
        [1, 1, 3, 5], [4, 1, 2, 3],
        ComplexF64[2 + im, -1, 3 - im, 4], 5, 4
    )
    for index_type in (Int32, Int64)
        CR = csr_matrix(R; index_type = index_type)
        @test size(CR) == size(R)
        @test isapprox(Matrix(KAPreconditioners.sparse_matrix(CR)), Matrix(R))
        @test all(i -> issorted(CR.colval[nzrange(CR, i)]), 1:size(R, 1))
    end
end

@testset "Jutul preconditioner wrappers" begin
    A = poisson_2d(6)
    b = ones(size(A, 1))
    context = DefaultContext()

    @test AMGPreconditioner().options.smoother isa SPAI0
    @test AMGPreconditioner().options.interpolation isa ExtendedIInterpolation
    @test AMGPreconditioner(:aggregation).options.interpolation isa ConstantInterpolation
    @test AMGPreconditioner(:ruge_stuben).options.interpolation isa ClassicalInterpolation

    vendor = KASmootherPreconditioner(:vendor_ilu)
    @test vendor.config isa VendorILU
    @test_throws ArgumentError Jutul.update_preconditioner!(
        vendor, A, b, context, nothing
    )

    for preconditioner in (
            AMGPreconditioner(:ruge_stuben; coarse_size = 10),
            KASmootherPreconditioner(:spai0),
            KASmootherPreconditioner(:gauss_seidel),
            KASmootherPreconditioner(:ilu0),
            KASmootherPreconditioner(:dilu),
        )
        Jutul.update_preconditioner!(preconditioner, A, b, context, nothing)
        x = zeros(size(A, 1))
        Jutul.apply!(x, preconditioner, b)
        @test norm(b - A * x) < norm(b)

        B = copy(A)
        nonzeros(B) .*= 1.1
        Jutul.partial_update_preconditioner!(
            preconditioner, B, b, context, nothing
        )
        fill!(x, 0.0)
        Jutul.apply!(x, preconditioner, b)
        @test norm(b - B * x) < norm(b)
    end
end

@testset "AMG wrapper reuse modes refresh smoothers" begin
    A = poisson_2d(12)
    b = ones(size(A, 1))
    context = DefaultContext()
    preconditioner = AMGPreconditioner(
        :ruge_stuben;
        smoother_type = :ilu0,
        coarse_size = 10,
        reuse = :none,
        reuse_partial = :operators
    )
    Jutul.update_preconditioner!(preconditioner, A, b, context, nothing)

    hierarchy = preconditioner.factor
    old_levels = copy(hierarchy.levels)
    old_smoother_values = [copy(level.smoother.factors) for level in old_levels]
    B = copy(A)
    nonzeros(B) .*= 1.1
    Jutul.partial_update_preconditioner!(
        preconditioner, B, b, context, nothing
    )

    @test preconditioner.factor === hierarchy
    @test all(
        hierarchy.levels[i].smoother === old_levels[i].smoother
            for i in eachindex(old_levels)
    )
    @test all(
        old_smoother_values[i] != hierarchy.levels[i].smoother.factors
            for i in eachindex(old_levels)
    )

    partial_levels = copy(hierarchy.levels)
    Jutul.update_preconditioner!(preconditioner, A, b, context, nothing)
    @test preconditioner.factor === hierarchy
    @test all(
        hierarchy.levels[i].smoother !== partial_levels[i].smoother
            for i in eachindex(partial_levels)
    )
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
        size(ordinary, 1), size(ordinary, 2); backend = static_backend
    )
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
    @test norm(b - A * x) < norm(b)

    cpu_minbatch = 7
    threshold_matrix = csr_matrix(
        copy(ordinary.rowptr), copy(ordinary.colval), copy(ordinary.nzval),
        size(ordinary, 1), size(ordinary, 2);
        backend = KernelAbstractions.CPU(), block_size = cpu_minbatch
    )
    threshold_hierarchy = setup_amg(
        threshold_matrix, AMGOptions(coarse_size = 4)
    )
    @test threshold_hierarchy.block_size == cpu_minbatch
    @test all(
        minbatch(level.A) == cpu_minbatch
            for level in threshold_hierarchy.levels
    )
    @test all(
        level.smoother.block_size == cpu_minbatch
            for level in threshold_hierarchy.levels
    )
end

@testset "Extended+i truncation preserves row sums" begin
    A = sparse(
        [1, 1, 1, 1, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 2, 3, 4, 5],
        [20.0, -1.0, -2.0, -3.0, -4.0, 1.0, 1.0, 1.0, 1.0],
        5, 5
    )
    C = csr_matrix(A)
    cf = Int8[-1, 1, 1, 1, 1]
    cmap = Int32[0, 1, 2, 3, 4]
    strong = falses(length(C.nzval))
    for k in nzrange(C, 1)
        strong[k] = C.colval[k] != 1
    end
    full = KAPreconditioners.build_prolongation(
        C, cf, cmap, 4, strong, ExtendedIInterpolation(0.0, 4, 2, false)
    )
    truncated = KAPreconditioners.build_prolongation(
        C, cf, cmap, 4, strong, ExtendedIInterpolation(0.0, 2, 2, true)
    )
    full_row = full.nzval[full.rowptr[1]:(full.rowptr[2] - 1)]
    truncated_row = truncated.nzval[
        truncated.rowptr[1]:(truncated.rowptr[2] - 1),
    ]
    @test length(truncated_row) == 2
    @test sum(full_row) ≈ 0.5
    @test sum(truncated_row) ≈ sum(full_row)
end

@testset "AMG interpolation tuning defaults" begin
    @test ClassicalInterpolation().norm_p == ExtendedIInterpolation().norm_p == 1
    @test ClassicalInterpolation().rescale
    @test ExtendedIInterpolation(0.0, 0).max_elements == 0
    @test AMGPreconditioner(:ruge_stuben).options.coarsening.theta == RugeStuben().theta
    @test AMGPreconditioner(:hmis).options.coarsening.theta == HMIS().theta
    @test !AMGPreconditioner(:ruge_stuben; second_pass = false).options.coarsening.second_pass
    @test_throws ArgumentError AMGPreconditioner(:hmis; second_pass = false)

    # A factor of 0.3 retains the weight 0.4 relative to a maximum of 1.
    A = sparse([1, 1, 1, 2, 3], [1, 2, 3, 2, 3], [10.0, -1.0, -0.4, 1.0, 1.0], 3, 3)
    C = csr_matrix(A)
    cf, cmap = Int8[-1, 1, 1], Int32[0, 1, 2]
    strong = KAPreconditioners.strength(C, 0.25, 1.0)
    for interpolation in (ClassicalInterpolation, ExtendedIInterpolation)
        P = KAPreconditioners.build_prolongation(C, cf, cmap, 2, strong, interpolation(0.3))
        @test P.rowptr[2] - P.rowptr[1] == 2
        # Explicit squared-magnitude truncation retains the previous semantics.
        P2 = KAPreconditioners.build_prolongation(C, cf, cmap, 2, strong, interpolation(0.3, 0, 2))
        @test P2.rowptr[2] - P2.rowptr[1] == 1
        @test sum(P2.nzval[P2.rowptr[1]:(P2.rowptr[2] - 1)]) ≈ 0.14
    end
end

function test_numeric_interpolation_tuning(backend)
    # Include a strong F-F edge, weak entries, and a distance-two-only coarse
    # point. The correct untruncated row sums are deliberately below one.
    A = sparse(
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 6, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6],
        [10.0, -2.0, -1.0, -1.5, -0.1, -1.0, 4.0, -0.5, -0.75, -2.0, 0.25, 1.0, 1.0, 1.0, 1.0],
        6, 6
    )
    cf, cmap = Int8[-1, -1, 1, 1, 1, 1], Int32[0, 0, 1, 2, 3, 4]
    C = csr_matrix(A)
    strong = KAPreconditioners.strength(C, 0.25, 1.0)
    for interpolation in (ClassicalInterpolation, ExtendedIInterpolation), rescale in (false, true)
        config = interpolation(0.0, 1, 1, rescale)
        P = KAPreconditioners.build_prolongation(C, cf, cmap, 4, strong, config)
        D = csr_matrix(A; backend = backend)
        dcf = KAPreconditioners.backend_copy(backend, cf)
        dmap = KAPreconditioners.backend_copy(backend, cmap)
        ds = KAPreconditioners.backend_copy(backend, strong)
        dp = KAPreconditioners.backend_copy(backend, P.nzval)
        drp = KAPreconditioners.backend_copy(backend, P.rowptr)
        dcv = KAPreconditioners.backend_copy(backend, P.colval)
        k! = KAPreconditioners.update_interpolation_p_kernel!(backend, 128)
        k!(dp, drp, dcv, D.rowptr, D.colval, D.nzval, dcf, dmap, ds,
            config isa ExtendedIInterpolation, rescale, 6; ndrange = 6)
        KAPreconditioners.synchronize_backend(backend)
        @test Array(dp) ≈ P.nzval

        # Change coefficients while retaining the old interpolation pattern.
        B = copy(A)
        B[1, 1] = 12.0
        B[1, 3] = -1.2
        B[2, 5] = -2.5
        BC = csr_matrix(B)
        full = KAPreconditioners.build_prolongation(BC, cf, cmap, 4, strong, interpolation(0.0, 0, 1, false))
        expected = copy(P.nzval)
        for i in 1:6
            full_row = full.rowptr[i]:(full.rowptr[i + 1] - 1)
            kept_row = P.rowptr[i]:(P.rowptr[i + 1] - 1)
            for pidx in kept_row
                q = findfirst(q -> full.colval[q] == P.colval[pidx], full_row)
                expected[pidx] = full.nzval[full_row[q]]
            end
            if rescale && !isempty(kept_row)
                expected[kept_row] .*= sum(full.nzval[full_row]) / sum(expected[kept_row])
            end
        end
        DB = csr_matrix(B; backend = backend)
        k!(dp, drp, dcv, DB.rowptr, DB.colval, DB.nzval, dcf, dmap, ds,
            config isa ExtendedIInterpolation, rescale, 6; ndrange = 6)
        KAPreconditioners.synchronize_backend(backend)
        @test Array(dp) ≈ expected
    end
end

@testset "Interpolation rescaling during numeric updates" begin
    test_numeric_interpolation_tuning(CPU())
    test_numeric_interpolation_tuning(JLBackend())
end

@testset "AMG strength selection" begin
    A = sparse([1, 1, 1, 2, 2, 3], [1, 2, 3, 2, 3, 3], [5.0, -1.0, 2.0, 3.0, 1.0, 1.0], 3, 3)
    C = csr_matrix(A)
    for backend in (CPU(), JLBackend()), mode in (:signed_fallback, :signed, :absolute)
        D = csr_matrix(A; backend = backend)
        strong = Array(KAPreconditioners.strength(D, 0.25, 1.0; strength_type = mode))
        expected = KAPreconditioners.strength(C, 0.25, 1.0; strength_type = mode)
        @test strong == expected
        by_edge = Dict((i, Int(C.colval[k])) => strong[k] for i in 1:3 for k in nzrange(C, i))
        @test by_edge[(1, 2)]
        @test by_edge[(1, 3)] == (mode == :absolute)
        @test by_edge[(2, 3)] == (mode != :signed)
    end
    @test AMGPreconditioner(; strength_type = :absolute).options.strength_type == :absolute
    @test_throws ArgumentError setup_amg(A, AMGOptions(strength_type = :invalid))
end

@testset "Jutul simulation with KA AMG" begin
    grid = CartesianMesh((3, 3), (1.0, 1.0))
    model = SimulationModel(DiscretizedDomain(grid), SimpleHeatSystem())
    initial_state = setup_state(model, Dict(:T => collect(range(0.0, 1.0; length = 9))))
    simulator = Simulator(model; state0 = initial_state)
    linear_solver = GenericKrylov(
        :bicgstab;
        preconditioner = AMGPreconditioner(
            :aggregation;
            coarse_size = 4
        )
    )
    states, = simulate(simulator, [1.0]; linear_solver, info_level = -1)
    @test length(states) == 1
end

@testset "coarsening and pure cycles" begin
    A = poisson_2d(14)
    b = ones(size(A, 1))
    configurations = (
        AMGOptions(coarsening = Aggregation(0.25), coarse_size = 12),
        AMGOptions(coarsening = RugeStuben(0.25), coarse_size = 12),
        AMGOptions(
            coarsening = HMIS(0.5),
            interpolation = ExtendedIInterpolation(0.0, 4, 2, false),
            coarse_size = 12
        ),
    )
    @test configurations[1].interpolation isa ConstantInterpolation
    @test configurations[2].interpolation isa ClassicalInterpolation
    @test configurations[3].interpolation isa ExtendedIInterpolation
    for options in configurations
        H = setup_amg(A, options)
        @test length(H.levels) >= 2
        @test H.levels[end].coarse_solver isa KAPreconditioners.CoarseLUState
        x = zeros(size(A, 1))
        r0 = norm(b - A * x)
        for _ in 1:4
            cycle!(x, H, b)
        end
        @test norm(b - A * x) < r0
    end
end

@testset "Ruge-Stuben splitting invariants" begin
    # A directed strength graph must not make a point fine merely because a
    # coarse point depends on it. Every F point needs an outgoing strong path
    # to a C point for classical interpolation.
    directed = sparse(
        [1, 1, 2, 3, 3, 4, 4],
        [1, 2, 2, 1, 3, 1, 4],
        [2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0], 4, 4
    )
    C = csr_matrix(directed)
    strong = KAPreconditioners.strength(C, 0.25, 1.0)
    cf, cmap, nc = KAPreconditioners.cf_split(C, strong, RugeStuben())
    P = KAPreconditioners.build_prolongation(
        C, cf, cmap, nc, strong, ClassicalInterpolation()
    )
    @test all(i -> P.rowptr[i + 1] > P.rowptr[i], eachindex(cf))
    @test all(eachindex(cf)) do i
        cf[i] == 1 || any(nzrange(C, i)) do k
            strong[k] && cf[C.colval[k]] == 1
        end
    end

    # The second RS pass enforces C2: strongly connected F points share a
    # direct strong C neighbor. The first pass alone violates C2 on this graph.
    edges = [(1, 3), (1, 5), (1, 7), (2, 4), (2, 6), (2, 8), (3, 4)]
    rows, cols, values = collect(1:8), collect(1:8), fill(4.0, 8)
    for (i, j) in edges
        push!(rows, i, j)
        push!(cols, j, i)
        push!(values, -1.0, -1.0)
    end
    C = csr_matrix(sparse(rows, cols, values, 8, 8))
    strong = KAPreconditioners.strength(C, 0.25, 1.0)
    first_cf, _, first_nc = KAPreconditioners.cf_split(C, strong, RugeStuben(; second_pass = false))
    @test first_nc == 2
    @test first_cf[3] == first_cf[4] == -1
    cf, _, nc = KAPreconditioners.cf_split(C, strong, RugeStuben())
    @test nc == 3
    hmis = setup_amg(C, AMGOptions(coarsening = HMIS(0.25), coarse_size = 2, max_row_sum = 1.0))
    @test hmis.levels[1].P.ncol == 2
    classical_hmis = setup_amg(C, AMGOptions(
        coarsening = HMIS(0.25), interpolation = ClassicalInterpolation(),
        coarse_size = 2, max_row_sum = 1.0
    ))
    @test classical_hmis.levels[1].cf == cf
    for i in eachindex(cf)
        cf[i] == -1 || continue
        coarse = Set(
            C.colval[k] for k in nzrange(C, i)
                if strong[k] && cf[C.colval[k]] == 1
        )
        for k in nzrange(C, i)
            j = C.colval[k]
            strong[k] && cf[j] == -1 || continue
            @test any(nzrange(C, j)) do q
                strong[q] && C.colval[q] in coarse
            end
        end
    end
end

@testset "Aggressive path counts and interpolation controls" begin
    # C1 reaches C4 through two distinct F points and C5 directly. HYPRE
    # gives the direct edge weight two in its path-count graph.
    A = sparse([1, 1, 1, 2, 3], [2, 3, 5, 4, 4], fill(-1.0, 5), 5, 5)
    C = csr_matrix(A)
    cf, cmap = Int8[1, -1, -1, 1, 1], Int32[1, 0, 0, 2, 3]
    for paths in (1, 2)
        graph, _ = KAPreconditioners.second_strength_graph(C, trues(5), cf, cmap, 3, paths)
        @test graph.colval == Int32[2, 3]
    end
    graph, _ = KAPreconditioners.second_strength_graph(C, trues(5), cf, cmap, 3, 3)
    @test isempty(graph.colval)

    A = poisson_2d(14)
    aggressive_interp = ExtendedIInterpolation(0.3, 2)
    options = AMGOptions(aggressive_levels = 1, aggressive_num_paths = 2,
        aggressive_interpolation = aggressive_interp, coarse_size = 8)
    @test KAPreconditioners.interpolation_for_level(options, 1) === aggressive_interp
    @test KAPreconditioners.interpolation_for_level(options, 2) === options.interpolation
    inherited = AMGOptions(coarsening = RugeStuben(), aggressive_levels = 1)
    @test KAPreconditioners.interpolation_for_level(inherited, 1) isa TwoStageExtendedIInterpolation
    @test KAPreconditioners.interpolation_for_level(inherited, 1).final.max_elements == 0
    wrapper = AMGPreconditioner(; aggressive_levels = 1, aggressive_num_paths = 2,
        aggressive_interpolation = aggressive_interp)
    @test wrapper.options.aggressive_num_paths == 2
    @test wrapper.options.aggressive_interpolation === aggressive_interp
    for backend in (CPU(), JLBackend())
        D = csr_matrix(A; backend = backend)
        H = setup_amg(D, options)
        P = H.levels[1].P
        @test maximum(diff(Array(P.rowptr))) <= 2
        previous = Array(P.nzval)
        resetup_amg!(H, D, :sparsity)
        @test Array(P.nzval) ≈ previous
        test_galerkin(H)
        b = KAPreconditioners.backend_copy(backend, ones(size(A, 1)))
        x = KAPreconditioners.backend_zeros(backend, Float64, size(A, 1))
        for _ in 1:4
            cycle!(x, H, b)
        end
        @test norm(ones(size(A, 1)) - A * Array(x)) < sqrt(size(A, 1))
    end
    @test_throws ArgumentError setup_amg(A, AMGOptions(aggressive_num_paths = 0))
    @test_throws ArgumentError setup_amg(A, AMGOptions(aggressive_interpolation = ClassicalInterpolation()))
    @test_throws ArgumentError setup_amg(A, AMGOptions(coarsening = Aggregation(), aggressive_num_paths = 2))
    @test_throws ArgumentError setup_amg(A, AMGOptions(coarsening = Aggregation(), aggressive_interpolation = aggressive_interp))
end

function test_two_stage_interpolation(backend)
    # A seven-point chain gives C1={2,4,6}, C2={4}. P2 uses the original
    # rows 2 and 6: each has effective diagonal 2.5 and numerator -0.5.
    # Consequently the endpoints receive nonzero distance-three weights.
    A = spdiagm(-1 => fill(-1.0, 6), 0 => fill(4.0, 7), 1 => fill(-1.0, 6))
    options = AMGOptions(coarsening = RugeStuben(), aggressive_levels = 1,
        aggressive_interpolation = TwoStageExtendedIInterpolation(), coarse_size = 1,
        max_levels = 2)
    H = setup_amg(csr_matrix(A; backend), options)
    level = H.levels[1]
    plan = level.aggressive
    @test Array(plan.rows) == [2, 4, 6]
    @test findall(==(1), Array(level.cf)) == [4]
    @test Array(level.P.nzval) ≈ [0.05, 0.2, 0.3, 1.0, 0.3, 0.2, 0.05]
    @test sparse_prolongation(level.P) ≈ sparse_prolongation(plan.P1) * sparse_prolongation(plan.P2)

    # Uniform scaling must preserve strength and interpolation even below eps.
    for scale in (1.0e-20, 1.0e20)
        scaled = setup_amg(csr_matrix(scale*A; backend), options)
        @test Array(scaled.levels[1].strength) == Array(level.strength)
        @test Array(scaled.levels[1].P.nzval) ≈ Array(level.P.nzval)
        resetup_amg!(H, csr_matrix(scale*A; backend), :sparsity)
        @test Array(level.P.nzval) ≈ Array(scaled.levels[1].P.nzval)
    end

    # Nonuniform coefficient changes exercise both factors and final-product
    # truncation. With the same split/stencil, reuse must match a fresh build.
    n = 12
    T = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
    E = spdiagm(-1 => fill(-1.0, n - 1), 1 => fill(-1.0, n - 1))
    A = kron(sparse(I, n, n), T) + kron(E, sparse(I, n, n))
    B = A + spdiagm(0 => [0.2 + 0.1*sin(i) for i in 1:size(A, 1)])
    for rescale in (false, true)
        interpolation = TwoStageExtendedIInterpolation(max_elements = 2,
            stage_max_elements = 2, rescale = rescale)
        options = AMGOptions(aggressive_levels = 1, aggressive_interpolation = interpolation,
            coarse_size = 1, max_levels = 2)
        H = setup_amg(csr_matrix(A; backend), options)
        fresh = setup_amg(csr_matrix(B; backend), options)
        P = H.levels[1].P
        pattern = (copy(Array(P.rowptr)), copy(Array(P.colval)))
        before = copy(Array(P.nzval))
        resetup_amg!(H, csr_matrix(B; backend), :sparsity)
        @test (Array(P.rowptr), Array(P.colval)) == pattern
        @test !(Array(P.nzval) ≈ before)
        # Candidate ranking may change, so compare the retained entries with
        # the complete product; preserve its full row sum when requested.
        full = sparse_prolongation(fresh.levels[1].aggressive.product)
        entries, all_columns, all_values = Array(P.rowptr), Array(P.colval), Array(P.nzval)
        for i in 1:P.nrow
            r = entries[i]:(entries[i + 1] - 1)
            columns = all_columns[r]
            expected = vec(Array(full[i, columns]))
            if rescale && !iszero(sum(expected))
                expected *= sum(full[i, :]) / sum(expected)
            end
            @test all_values[r] ≈ expected
        end
        @test KAPreconditioners.sparse_matrix(H.levels[2].A) ≈
            sparse_prolongation(P)' * B * sparse_prolongation(P)
        old_plan = H.levels[1].aggressive
        resetup_amg!(H, csr_matrix(B; backend), :memory)
        new_plan = H.levels[1].aggressive
        @test new_plan.P1.nzval === old_plan.P1.nzval
        @test new_plan.P2.nzval === old_plan.P2.nzval
        @test Array(H.levels[1].P.nzval) ≈ Array(fresh.levels[1].P.nzval)
    end
end

@testset "Hypre two-stage aggressive interpolation" begin
    @test_throws ArgumentError TwoStageExtendedIInterpolation(stage_max_elements = -1)
    @test_throws ArgumentError setup_amg(poisson_2d(3),
        AMGOptions(interpolation = TwoStageExtendedIInterpolation()))
    for backend in (CPU(), JLBackend())
        test_two_stage_interpolation(backend)
    end
end

@testset "aggressive coarsening" begin
    A = poisson_2d(20)
    regular_options = AMGOptions(coarse_size = 12)
    explicit_default = AMGOptions(coarse_size = 12, aggressive_levels = 0)
    aggressive_options = AMGOptions(
        coarse_size = 12, aggressive_levels = 1
    )
    regular = setup_amg(A, regular_options)
    defaulted = setup_amg(A, explicit_default)
    aggressive = setup_amg(A, aggressive_options)

    @test AMGOptions().aggressive_levels == 0
    @test regular.levels[1].cf == defaulted.levels[1].cf
    @test regular.levels[1].P.rowptr == defaulted.levels[1].P.rowptr
    @test regular.levels[1].P.colval == defaulted.levels[1].P.colval
    @test aggressive.levels[1].P.ncol < regular.levels[1].P.ncol
    @test all(i -> aggressive.levels[1].P.rowptr[i + 1] >
        aggressive.levels[1].P.rowptr[i], 1:size(A, 1))

    two_levels = setup_amg(
        A, AMGOptions(coarse_size = 12, aggressive_levels = 2)
    )
    @test two_levels.levels[2].P.ncol < aggressive.levels[2].P.ncol

    for coarsening in (RugeStuben(0.25), Aggregation(0.25))
        standard = setup_amg(
            A, AMGOptions(coarsening = coarsening, coarse_size = 12)
        )
        coarsened = setup_amg(
            A, AMGOptions(
                coarsening = coarsening, coarse_size = 12,
                aggressive_levels = 1
            )
        )
        @test coarsened.levels[1].P.ncol < standard.levels[1].P.ncol
    end

    b = ones(size(A, 1))
    x = zeros(size(A, 1))
    for _ in 1:4
        cycle!(x, aggressive, b)
    end
    @test norm(b - A * x) < norm(b)

    B = copy(A)
    B[1, 1] *= 1.2
    B[2, 2] *= 0.8
    @test resetup_amg!(aggressive, B, :sparsity) === aggressive
    test_galerkin(aggressive)
    @test resetup_amg!(aggressive, A, :memory) === aggressive
    @test aggressive.levels[1].P.ncol < regular.levels[1].P.ncol

    device = setup_amg(
        csr_matrix(A; backend = JLBackend()), aggressive_options
    )
    @test device.levels[1].P.ncol == aggressive.levels[1].P.ncol
    device_x = JLArray(zeros(size(A, 1)))
    device_b = JLArray(b)
    for _ in 1:4
        cycle!(device_x, device, device_b)
    end
    @test norm(b - A * Array(device_x)) < norm(b)

    wrapper = AMGPreconditioner(; aggressive_levels = 1, coarse_size = 12)
    @test wrapper.options.aggressive_levels == 1
    @test_throws ArgumentError setup_amg(
        A, AMGOptions(aggressive_levels = -1)
    )
end

@testset "coarse LU" begin
    @test AMGOptions().coarse_solver == :lu
    @test AMGOptions().coarse_size == 50
    M = [0.0 2.0 1.0; 1.0 1.0 0.0; 2.0 0.0 1.0]
    rhs = [1.0, -2.0, 3.0]
    factorization = lu!(copy(M))
    solution = similar(rhs)
    ldiv!(solution, factorization, rhs)
    @test M * solution ≈ rhs rtol = 1.0e-13 atol = 1.0e-13

    A = poisson_2d(10)
    H = setup_amg(A, AMGOptions(coarse_size = 10))
    b = ones(size(A, 1))
    x = zeros(size(A, 1))
    KAPreconditioners.apply!(x, H, b)
    fine = H.levels[end - 1]
    coarse = H.levels[end]
    @test Array(coarse.A * fine.correction) ≈ Array(fine.rhs) rtol = 1.0e-12 atol = 1.0e-12

    Hs = setup_amg(A, AMGOptions(coarse_size = 10, coarse_solver = :spai0))
    @test isnothing(Hs.levels[end].coarse_solver)
    @test_throws ArgumentError setup_amg(A, AMGOptions(coarse_solver = :invalid))
end

@testset "asymmetric Galerkin product" begin
    A = poisson_2d(8)
    A[1, 2] = -0.2
    A[2, 1] = -1.8
    A[10, 11] = -0.35
    A[11, 10] = -1.65
    H = setup_amg(A, AMGOptions(coarse_size = 8, max_row_sum = 0.9))
    test_galerkin(H)
end

@testset "in-place resetup" begin
    A = poisson_2d(12)
    H = setup_amg(A, AMGOptions(coarsening = HMIS(0.5), coarse_size = 10))
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
    @test Array(H.levels[end].A * H.levels[end - 1].correction) ≈
        Array(H.levels[end - 1].rhs) rtol = 1.0e-12 atol = 1.0e-12
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
    options = AMGOptions(coarsening = HMIS(0.5), coarse_size = 10)
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
    @test isapprox(x, reference; rtol = 1.0e-12, atol = 1.0e-12)

    # Level 1 is fixed by the discretization even though all coarse symbolic
    # data is rebuilt in :memory mode.
    level1_ids = (
        objectid(H.levels[1].A.rowptr),
        objectid(H.levels[1].A.colval), objectid(H.levels[1].A.nzval),
    )
    resetup_amg!(H, A, :memory)
    @test level1_ids == (
        objectid(H.levels[1].A.rowptr),
        objectid(H.levels[1].A.colval), objectid(H.levels[1].A.nzval),
    )
    @test isapprox(Matrix(KAPreconditioners.sparse_matrix(H.levels[1].A)), Matrix(A))
    test_galerkin(H)

    changed_graph = copy(A)
    changed_graph[1, end] = -0.05
    @test_throws ArgumentError resetup_amg!(H, changed_graph, :memory)
end

@testset "backend buffer capacity reuse" begin
    backend = JLBackend()
    allocation = JLArray(collect(Int32, 1:12))
    shortened = KAPreconditioners.copy_reusing(
        allocation, collect(Int32, 1:7), backend
    )
    @test length(shortened) == 7
    @test KAPreconditioners.reusable_buffer(shortened) === allocation
    @test Array(shortened) == collect(Int32, 1:7)

    regrown = KAPreconditioners.copy_reusing(
        shortened, collect(Int32, 8:-1:1), backend
    )
    @test length(regrown) == 8
    @test KAPreconditioners.reusable_buffer(regrown) === allocation
    @test Array(regrown) == collect(Int32, 8:-1:1)

    tracker = Ref(0)
    expanded = KAPreconditioners.copy_reusing(
        regrown, collect(Int32, 1:16), backend;
        reallocation_tracker = tracker
    )
    expanded_allocation = KAPreconditioners.reusable_buffer(expanded)
    @test tracker[] == sizeof(Int32) * length(allocation)
    @test length(expanded) == 16
    @test length(expanded_allocation) > length(expanded)
    @test Array(expanded) == collect(Int32, 1:16)

    tracker[] = 0
    reused_expansion = KAPreconditioners.copy_reusing(
        expanded, collect(Int32, 17:-1:1), backend;
        reallocation_tracker = tracker
    )
    @test iszero(tracker[])
    @test KAPreconditioners.reusable_buffer(reused_expansion) ===
        expanded_allocation
    @test Array(reused_expansion) == collect(Int32, 17:-1:1)

    tracker[] = 0
    zero_buffer = KAPreconditioners.zeros_reusing(
        JLArray(zeros(Float64, 4)), backend, Float64, 7;
        reallocation_tracker = tracker
    )
    zero_allocation = KAPreconditioners.reusable_buffer(zero_buffer)
    @test tracker[] == sizeof(Float64) * 4
    @test length(zero_allocation) > length(zero_buffer)
    @test all(iszero, Array(zero_buffer))

    tracker[] = 0
    reused_zero_buffer = KAPreconditioners.zeros_reusing(
        zero_buffer, backend, Float64, 6;
        reallocation_tracker = tracker
    )
    @test iszero(tracker[])
    @test KAPreconditioners.reusable_buffer(reused_zero_buffer) ===
        zero_allocation

    large_matrix = csr_matrix(spdiagm(0 => fill(2.0, 12)); backend = backend)
    small_matrix = csr_matrix(spdiagm(0 => fill(2.0, 7)); backend = backend)
    smoother = setup_smoother(large_matrix, SPAI0())
    tracker[] = 0
    setup_smoother(
        small_matrix, SPAI0(); reuse = smoother,
        reallocation_tracker = tracker
    )
    @test iszero(tracker[])

end

@testset "standalone and Krylov solves" begin
    A = poisson_2d(18)
    b = ones(size(A, 1))
    H = setup_amg(A, AMGOptions(coarse_size = 12))
    x = zeros(size(A, 1))
    x, iterations = solve!(x, H, b; rtol = 1.0e-7, maxiter = 60)
    @test iterations <= 60
    @test norm(b - A * x) / norm(b) < 1.0e-6

    Hk = setup_amg(A, AMGOptions(coarse_size = 12))
    # Krylov's left-preconditioned stopping norm is not the true residual, so
    # request a tighter internal tolerance before checking the latter.
    xk, stats = gmres(A, b; M = Hk, ldiv = true, rtol = 1.0e-10, itmax = 100)
    @test norm(b - A * xk) / norm(b) < 1.0e-7
    @test stats.niter < 100
end

@testset "input validation" begin
    A = poisson_2d(4)
    H = setup_amg(A)
    @test_throws ArgumentError resetup_amg!(H, A, :invalid)
    @test_throws DimensionMismatch setup_amg(sparse(ones(3, 2)))
    @test_throws ArgumentError setup_amg(A, AMGOptions(max_row_sum = -0.1))
    @test_throws ArgumentError setup_amg(
        A, AMGOptions(
            coarsening = Aggregation(), interpolation = ClassicalInterpolation()
        )
    )
    @test_throws ArgumentError setup_amg(
        A, AMGOptions(
            coarsening = RugeStuben(), interpolation = ConstantInterpolation()
        )
    )
end

@testset "max row sum" begin
    A = sparse(
        [1, 1, 2, 2, 2, 3, 3],
        [1, 2, 1, 2, 3, 2, 3],
        [2.0, -0.1, -1.0, 2.0, -1.0, -1.0, 2.0], 3, 3
    )
    C = csr_matrix(A)
    original = copy(C.nzval)
    regular = Array(KAPreconditioners.strength(C, 0.25, 1.0))
    weakened = Array(KAPreconditioners.strength(C, 0.25, 0.9))
    row1 = C.rowptr[1]:(C.rowptr[2] - 1)
    @test any(regular[row1])
    @test !any(weakened[row1])
    @test C.nzval == original
    @test AMGOptions(max_row_sum = 0.9).max_row_sum == 0.9
end

@testset "HMIS and Extended+i semantics" begin
    F = sparse([1, 1, 1, 2, 3], [1, 2, 3, 2, 3], [2.0, 1.0, 0.5, 1.0, 1.0], 3, 3)
    CF = csr_matrix(F)
    fallback_strength = KAPreconditioners.strength(CF, 0.5, 1.0)
    row1 = CF.rowptr[1]:(CF.rowptr[2] - 1)
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
    P = KAPreconditioners.build_prolongation(
        CW, Int8[-1, 1], Int32[0, 1], 1, weak,
        ExtendedIInterpolation(0.0, 4, 2, true)
    )
    @test P.rowptr[1] == P.rowptr[2]
    @test P.rowptr[3] - P.rowptr[2] == 1
end

@testset "KernelAbstractions device hierarchy" begin
    A = poisson_2d(10)
    C = csr_matrix(A)
    backend = JLBackend()
    D = csr_matrix(A; backend = backend)
    @test D.rowptr isa JLArray
    @test Matrix(KAPreconditioners.sparse_matrix(D)) ≈ Matrix(A)
    H = setup_amg(D, AMGOptions(coarse_size = 10))
    @test H.levels[1].A.nzval isa JLArray
    @test all(level -> level.A.nzval isa JLArray, H.levels)
    coarse_solver = H.levels[end].coarse_solver
    @test coarse_solver isa SparseLU
    @test coarse_solver.factorization isa KAPreconditioners.KASparseLUFactor
    b = JLArray(ones(size(A, 1)))
    x = JLArray(zeros(size(A, 1)))
    for _ in 1:4
        cycle!(x, H, b)
    end
    @test norm(ones(size(A, 1)) - A * Array(x)) < norm(ones(size(A, 1)))

    D2 = csr_matrix(
        copy(D.rowptr), copy(D.colval), 1.2 .* D.nzval,
        size(D, 1), size(D, 2); backend = backend
    )
    ids = map(level -> objectid(level.A.nzval), H.levels)
    resetup_amg!(H, D2, :sparsity)
    @test ids == map(level -> objectid(level.A.nzval), H.levels)
    @test H.levels[end].coarse_solver === coarse_solver

    DB = csr_matrix(
        copy(D.rowptr), copy(D.colval),
        D.nzval .* JLArray(
            [
                if isodd(i)
                    0.8
                else
                    1.3
                end for i in eachindex(D.nzval)
            ]
        ),
        size(D, 1), size(D, 2); backend = backend
    )
    level1_ids = (
        objectid(H.levels[1].A.rowptr), objectid(H.levels[1].A.colval),
        objectid(H.levels[1].A.nzval),
    )
    resetup_amg!(H, DB, :memory)
    test_galerkin(H)
    @test H.levels[end].coarse_solver isa SparseLU
    @test level1_ids == (
        objectid(H.levels[1].A.rowptr), objectid(H.levels[1].A.colval),
        objectid(H.levels[1].A.nzval),
    )
    resetup_amg!(H, D, :memory)
    test_galerkin(H)
    @test length(H.levels) >= 3
    kept_ids = [level_memory_ids(level) for level in H.levels[1:2]]
    resetup_amg!(H, DB, :partial_sparsity; n_levels_partial_keep = 2)
    @test kept_ids == [level_memory_ids(level) for level in H.levels[1:2]]
    test_galerkin(H)
    resetup_amg!(H, D, :partial_operators; n_levels_partial_keep = 2)
    test_galerkin(H)

    # Exercise every method-specific interpolation reset on device arrays.
    for coarsening in (RugeStuben(), Aggregation())
        method_options = AMGOptions(coarsening = coarsening, coarse_size = 10)
        method_hierarchy = setup_amg(D, method_options)
        resetup_amg!(method_hierarchy, D2, :sparsity)
        method_x = JLArray(zeros(size(A, 1)))
        cycle!(method_x, method_hierarchy, b)
        @test all(isfinite, Array(method_x))
        test_galerkin(method_hierarchy)
    end
end

function reference_hybrid_gs(A, b, initial, config, partitions; steps = config.steps)
    x = copy(initial)
    n = length(x)
    base, extra = divrem(n, min(partitions, n))
    for _ in 1:steps
        frozen = copy(x)
        first = 1
        for p in 1:min(partitions, n)
            last = first + base + (p <= extra) - 1
            for rows in (first:last, last:-1:first), i in rows
                iszero(A[i, i]) && continue
                outside = b[i]
                inside, old_inside = zero(eltype(x)), zero(eltype(x))
                for j in 1:n
                    i == j && continue
                    if first <= j <= last
                        inside -= A[i, j] * x[j]
                        old_inside += A[i, j] * frozen[j]
                    else
                        outside -= A[i, j] * frozen[j]
                    end
                end
                w, omega = config.damping, config.omega
                x[i] = (1 - w*omega)*x[i] +
                    w*(omega*outside + inside + (1 - omega)*old_inside)/A[i, i]
            end
            first = last + 1
        end
    end
    return x
end

function test_hybrid_gauss_seidel(backend)
    # Unequal partition sizes, signed and zero diagonals, and asymmetric
    # cross-partition edges detect stale-value and backward-sweep errors.
    A = sparse([4.0 -1 0 0.3 0 0 0; -2 5 -1 0 0 0 0; 0 -1 3 -0.5 0 0 0;
        0 0.2 -1 -4 1 0 0; 0 0 0 -1 4 -1 0; 0 0 0 0 -2 5 1;
        -0.2 0 0 0 0 -1 0])
    rhs, initial = collect(1.0:7.0), sin.(collect(1.0:7.0))
    device(v) = KAPreconditioners.backend_copy(backend, v)
    C, b = csr_matrix(A; backend), device(rhs)
    for partitions in (1, 3, 9), (w, omega) in ((1.0, 1.0), (0.7, 1.0), (1.2, 0.8))
        config = HybridGaussSeidel(2, w; omega, partitions)
        state = setup_smoother(C, config)
        @test length(state.partition_offsets) - 1 == min(partitions, 7)
        x = device(initial)
        expected = reference_hybrid_gs(A, rhs, initial, config, partitions)
        smooth!(x, state, C, b)
        @test Array(x) ≈ expected
        @test Array(x)[7] == initial[7]
        # A precomputed residual must give exactly the same symmetric step.
        x = device(initial)
        r = device(rhs - A*initial)
        KAPreconditioners.smooth_level!(x, C, b, state, 2; residual = r)
        @test Array(x) ≈ expected
        # Zero-initial application must overwrite poisoned persistent buffers.
        fill!(state.forward, NaN)
        fill!(state.work, NaN)
        fill!(x, NaN)
        KAPreconditioners.apply!(x, state, b)
        @test Array(x) ≈ reference_hybrid_gs(A, rhs, zeros(7), config, partitions)
        inverse_id, lower_id = objectid(state.inverse_diagonal), objectid(state.lower_rows)
        native = state.native
        B = 1.3A
        for i in 1:6
            B[i, i] += 0.1*i
        end
        B[1, 2] *= 0.9
        D = csr_matrix(B; backend)
        @test setup_smoother(D, config; reuse = state) === state
        @test state.native === native
        @test objectid(state.inverse_diagonal) == inverse_id
        @test objectid(state.lower_rows) == lower_id
        fill!(x, NaN)
        KAPreconditioners.apply!(x, state, b)
        @test Array(x) ≈ reference_hybrid_gs(B, rhs, zeros(7), config, partitions)
        @test_throws ArgumentError update_smoother!(state, csr_matrix(spdiagm(0 => ones(7)); backend))
    end
    # One partition, unit weights: the SGS preconditioner is the symmetric
    # triangular product, independently of the smoother implementation.
    A = spdiagm(-1 => fill(-1.0, 6), 0 => fill(4.0, 7), 1 => fill(-1.0, 6))
    C = csr_matrix(A; backend)
    state = setup_smoother(C, HybridGaussSeidel(; partitions = 1))
    x = device(fill(NaN, 7))
    KAPreconditioners.apply!(x, state, b)
    @test Array(x) ≈ UpperTriangular(Matrix(A)) \ (Diagonal(diag(A)) * (LowerTriangular(Matrix(A)) \ rhs))
    auto = setup_smoother(C, HybridGaussSeidel())
    auto_partitions = length(auto.partition_offsets) - 1
    if backend isa CPU || backend isa JLBackend
        @test auto_partitions == min(7, backend isa CPU ? Threads.nthreads() : 1)
    end
    KAPreconditioners.apply!(x, auto, b)
    @test Array(x) ≈ reference_hybrid_gs(A, rhs, zeros(7), auto.config, auto_partitions)
    # Larger automatic partitions must retain weighted within-partition GS
    # and frozen cross-partition coupling, including after coefficient updates.
    n_auto = 257
    auto_matrix = spdiagm(-1 => fill(-1.0, n_auto-1),
        0 => 3 .+ collect(1:n_auto)./n_auto, 1 => fill(-0.5, n_auto-1))
    auto_matrix[1, 180], auto_matrix[185, 12] = 0.2, -0.3
    auto_rhs = sin.(collect(1.0:n_auto))
    auto_config = HybridGaussSeidel(2, 0.7; omega = 0.8)
    auto = setup_smoother(csr_matrix(auto_matrix; backend), auto_config)
    auto_partitions = length(auto.partition_offsets)-1
    auto_native = auto.native
    auto_x, auto_b = device(fill(NaN, n_auto)), device(auto_rhs)
    for shift in (0.0, 0.2)
        B = auto_matrix + spdiagm(0 => shift.*collect(1:n_auto)./n_auto)
        update_smoother!(auto, csr_matrix(B; backend))
        @test auto.native === auto_native
        KAPreconditioners.apply!(auto_x, auto, auto_b)
        @test Array(auto_x) ≈ reference_hybrid_gs(B, auto_rhs, zeros(n_auto),
            auto_config, auto_partitions)
    end
    # Float32 weights and work must stay Float32 on accelerators.
    single = setup_smoother(csr_matrix(Float32.(A); backend), HybridGaussSeidel(1, 0.8; partitions = 2))
    y = device(fill(Float32(NaN), 7))
    KAPreconditioners.apply!(y, single, device(Float32.(rhs)))
    @test eltype(single.work) === Float32
    @test Array(y) ≈ reference_hybrid_gs(Float32.(A), Float32.(rhs), zeros(Float32, 7),
        single.config, 2) rtol = 1.0e-6

    n = 6
    T = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
    E = spdiagm(-1 => fill(-1.0, n - 1), 1 => fill(-1.0, n - 1))
    A = kron(sparse(I, n, n), T) + kron(E, sparse(I, n, n))
    C = csr_matrix(A; backend)
    H = setup_amg(C, AMGOptions(smoother = HybridGaussSeidel(; partitions = 2),
        aggressive_levels = 1, coarse_size = 4))
    b, x = device(ones(n*n)), device(fill(NaN, n*n))
    KAPreconditioners.apply!(x, H, b)
    @test norm(ones(n*n) - A*Array(x)) < n
    # An executable cycle must agree with ordinary execution for changing RHS,
    # and retain its storage across purely numerical hierarchy updates.
    function check_execution(H, b)
        expected = similar(b)
        KAPreconditioners.vcycle!(expected, b, H, 1; residual = b, zero_initial = true)
        actual = similar(b)
        KAPreconditioners.apply!(actual, H, b)
        @test Array(actual) ≈ Array(expected)
    end
    check_execution(H, device(sin.(collect(1.0:n*n))))
    for mode in (:operators, :sparsity, :memory, :partial_sparsity)
        execution = H.execution
        B = copy(A)
        # Nonuniform coefficient changes exercise normalized values, the
        # interpolation/Galerkin operators, and renewed coarse pivoting.
        nonzeros(B) .*= 1 .+ 0.05 .* sin.(collect(1:length(nonzeros(B))))
        B += spdiagm(0 => fill(0.2, n*n))
        resetup_amg!(H, csr_matrix(B; backend), mode; n_levels_partial_keep = 1)
        if !isnothing(execution)
            @test H.execution === (mode in (:operators, :sparsity) ? execution : nothing)
        end
        fill!(x, NaN)
        KAPreconditioners.apply!(x, H, b)
        @test all(isfinite, Array(x))
        @test norm(ones(n*n) - B*Array(x)) < n
        check_execution(H, device(cos.(collect(1.0:n*n))))
    end
    if !isnothing(H.execution)
        # Stable graph buffers permit different vector allocations and aliasing
        # of the external input/output, without changing the captured pointers.
        rhs = device(ones(n*n))
        KAPreconditioners.apply!(rhs, H, rhs)
        @test Array(rhs) ≈ Array(x)
    end
end

@testset "Hypre hybrid symmetric GS/SSOR" begin
    @test_throws ArgumentError HybridGaussSeidel(0)
    @test_throws ArgumentError HybridGaussSeidel(1, 2.0)
    @test_throws ArgumentError HybridGaussSeidel(; omega = NaN)
    @test_throws ArgumentError HybridGaussSeidel(; partitions = -1)
    @test AMGPreconditioner(smoother_type = :hybrid_gauss_seidel).options.smoother isa HybridGaussSeidel
    @test KASmootherPreconditioner(:hybrid_ssor).config isa HybridGaussSeidel
    for backend in (CPU(), JLBackend())
        test_hybrid_gauss_seidel(backend)
    end
end

@testset "standalone ILU smoothers" begin
    A = poisson_2d(5)
    C = csr_matrix(A)
    b = ones(size(A, 1))

    for config in (ILU0(), DILU())
        state = setup_smoother(C, config)
        @test Any ∉ fieldtypes(typeof(state))
        @test size(state) == size(A)
        @test eltype(state) == Float64

        x = zeros(size(A, 1))
        KAPreconditioners.apply!(x, state, b)
        @test norm(b - A * x) < norm(b)
        @test state \ b ≈ x
        product = similar(x)
        mul!(product, state, b)
        @test product ≈ x

        fill!(x, 0.25)
        initial_residual = norm(b - A * x)
        smooth!(x, state, C, b; steps = 2)
        @test norm(b - A * x) < initial_residual

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
        @test setup_smoother(D, config; reuse = state) === state
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
        H = setup_amg(A, AMGOptions(smoother = config, coarse_size = 8))
        x = zeros(size(A, 1))
        cycle!(x, H, b)
        @test norm(b - A * x) < norm(b)
    end
end

@testset "ILU smoothers on KernelAbstractions arrays" begin
    A = poisson_2d(5)
    backend = JLBackend()
    C = csr_matrix(A; backend = backend)
    b = JLArray(ones(size(A, 1)))

    for config in (ILU0(), DILU())
        state = setup_smoother(C, config)
        @test Any ∉ fieldtypes(typeof(state))
        x = JLArray(zeros(size(A, 1)))
        KAPreconditioners.apply!(x, state, b)
        @test norm(ones(size(A, 1)) - A * Array(x)) < norm(ones(size(A, 1)))

        host_state = setup_smoother(csr_matrix(A), config)
        host_x = zeros(size(A, 1))
        KAPreconditioners.apply!(host_x, host_state, ones(size(A, 1)))
        @test Array(state.inverse_diagonal) ≈ host_state.inverse_diagonal
        @test Array(x) ≈ host_x

        fill!(x, 0.25)
        initial_residual = norm(ones(size(A, 1)) - A * Array(x))
        smooth!(x, state, C, b; steps = 2)
        @test norm(ones(size(A, 1)) - A * Array(x)) < initial_residual
    end
end

@testset "fused ILU post-smoothing" begin
    A = poisson_2d(5)
    initial = fill(0.25, size(A, 1))
    rhs = ones(size(A, 1))
    for backend in (KernelAbstractions.CPU(), JLBackend())
        C = csr_matrix(A; backend = backend)
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
            smooth!(reference, state, C, b; steps = 2)
            KAPreconditioners.smooth_result!(fused, C, b, state, 2)
            @test Array(fused) ≈ Array(reference)
        end
    end
end

@testset "DILU recurrence and scale invariance" begin
    A = sparse([4.0 1.0 0.0; 2.0 5.0 3.0; 0.0 4.0 6.0])
    d1 = inv(A[1, 1])
    d2 = inv(A[2, 2] - A[2, 1] * d1 * A[1, 2])
    d3 = inv(A[3, 3] - A[3, 2] * d2 * A[2, 3])
    expected_diagonal = [d1, d2, d3]
    rhs = [1.0, -2.0, 4.0]

    y1 = d1 * rhs[1]
    y2 = d2 * (rhs[2] - A[2, 1] * y1)
    y3 = d3 * (rhs[3] - A[3, 2] * y2)
    z3 = y3
    z2 = y2 - d2 * A[2, 3] * z3
    z1 = y1 - d1 * A[1, 2] * z2
    expected = [z1, z2, z3]

    for backend in (KernelAbstractions.CPU(), JLBackend())
        C = csr_matrix(A; backend = backend)
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
    scale = 1.0e-12
    state = setup_smoother(csr_matrix(A), DILU())
    scaled_state = setup_smoother(csr_matrix(scale * A), DILU())
    result = state \ rhs
    scaled_result = scaled_state \ rhs
    @test scaled_state.inverse_diagonal ≈ state.inverse_diagonal / scale
    @test scaled_result ≈ result / scale
end

@testset "static block smoothers" begin
    Block = SMatrix{2, 2, Float64, 4}
    BlockVector = SVector{2, Float64}
    diagonal = Block(4.0, 0.1, 0.2, 3.0)
    lower = Block(-1.0, 0.0, 0.1, -0.6)
    upper = Block(-0.8, -0.1, 0.0, -0.5)
    n = 8
    rows = vcat(collect(1:n), collect(2:n), collect(1:(n - 1)))
    columns = vcat(collect(1:n), collect(1:(n - 1)), collect(2:n))
    values = vcat(fill(diagonal, n), fill(lower, n - 1), fill(upper, n - 1))
    A = sparse(rows, columns, values, n, n)
    C = csr_matrix(A)
    b = [BlockVector(1.0 + 0.1 * i, -0.5 + 0.05 * i) for i in 1:n]

    for config in (SPAI0(), ILU0(), DILU())
        state = setup_smoother(C, config)
        x = fill(zero(BlockVector), n)
        KAPreconditioners.apply!(x, state, b)
        @test norm(b - A * x) < norm(b)

        fill!(x, BlockVector(0.2, -0.1))
        initial_residual = norm(b - A * x)
        smooth!(x, state, C, b; steps = 2)
        @test norm(b - A * x) < initial_residual
    end

    backend = JLBackend()
    D = csr_matrix(A; backend = backend)
    device_b = JLArray(b)
    for config in (ILU0(), DILU())
        state = setup_smoother(D, config)
        x = JLArray(fill(zero(BlockVector), n))
        KAPreconditioners.apply!(x, state, device_b)
        @test norm(b - A * Array(x)) < norm(b)

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
        @test ka_result ≈ serial_result rtol = 1.0e-12 atol = 1.0e-12
        return serial_result
    end

    scalar_matrix = poisson_2d(5)
    compare_ilu(scalar_matrix, ones(size(scalar_matrix, 1)))

    Block = SMatrix{2, 2, Float64, 4}
    BlockVector = SVector{2, Float64}
    diagonal = Block(4.0, 0.1, 0.2, 3.0)
    off_diagonal = Block(-0.8, -0.1, 0.0, -0.5)
    n = 8
    rows = vcat(1:n, 2:n, 1:(n - 1))
    columns = vcat(1:n, 1:(n - 1), 2:n)
    values = vcat(fill(diagonal, n), fill(off_diagonal, 2 * n - 2))
    block_matrix = sparse(rows, columns, values, n, n)
    rhs = [BlockVector(1.0 + 0.1 * i, -0.5 + 0.05 * i) for i in 1:n]
    block_reference = compare_ilu(block_matrix, rhs)

    flat_rhs = reinterpret(Float64, rhs)
    wrapped_smoother = KASmootherPreconditioner(:ilu0)
    Jutul.update_preconditioner!(
        wrapped_smoother, csr_matrix(block_matrix; index_type = Int),
        flat_rhs, DefaultContext(), nothing
    )
    @test Jutul.operator_nrows(wrapped_smoother) == length(flat_rhs)
    flat_result = similar(flat_rhs)
    Jutul.apply!(flat_result, wrapped_smoother, flat_rhs)
    @test isapprox(
        reinterpret(BlockVector, flat_result), block_reference;
        rtol = 1.0e-12, atol = 1.0e-12
    )

    device_matrix = csr_matrix(block_matrix; backend = JLBackend())
    device_rhs = JLArray(flat_rhs)
    device_result = similar(device_rhs)
    device_smoother = KASmootherPreconditioner(:ilu0)
    Jutul.update_preconditioner!(
        device_smoother, device_matrix,
        device_rhs, KernelAbstractionsContext(JLBackend()), nothing
    )
    Jutul.apply!(device_result, device_smoother, device_rhs)
    @test isapprox(
        reinterpret(BlockVector, Array(device_result)),
        block_reference; rtol = 1.0e-12, atol = 1.0e-12
    )
end
