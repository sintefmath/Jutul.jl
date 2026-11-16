using Jutul
using Jutul.KAPreconditioners
using LinearAlgebra
using SparseArrays
using Test

function check_pmis_galerkin(H)
    for l in 1:(length(H.levels) - 1)
        level = H.levels[l]
        A = KAPreconditioners.sparse_matrix(level.A)
        P = level.P
        rp, cv, pv = Array(P.rowptr), Array(P.colval), Array(P.nzval)
        Pt = SparseMatrixCSC(P.ncol, P.nrow, rp, cv, pv)
        coarse = KAPreconditioners.sparse_matrix(H.levels[l + 1].A)
        @test coarse ≈ Pt * A * transpose(Pt)
    end
end

function check_pmis_memory(backend, ::Type{T}, interpolation, coarse_solver) where {T}
    n = 12
    shift = spdiagm(-1 => fill(-one(T), n - 1), 1 => fill(-one(T), n - 1))
    id = sparse(I, n, n)
    Ax, Ay = kron(id, shift), kron(shift, id)
    # Flip the strong direction without changing any stored position.
    A = Ax + T(0.1) * Ay + T(3) * sparse(I, n^2, n^2)
    B = T(0.1) * Ax + Ay + T(3) * sparse(I, n^2, n^2)
    Adev = csr_matrix(A; backend, use_vendor_linalg = false)
    Bdev = csr_matrix(B; backend, use_vendor_linalg = false)
    opts = AMGOptions(; coarsening = PMIS(), interpolation, coarse_solver, coarse_size = 8)
    H = setup_amg(Adev, opts)
    @test length(H.levels) > 1
    check_pmis_galerkin(H)
    level = H.levels[1]
    reference_P = KAPreconditioners.build_prolongation(
        KAPreconditioners.host_csr(Adev), Array(level.cf), Array(level.coarse_map),
        level.P.ncol, Array(level.strength), interpolation
    )
    @test Array(level.P.rowptr) == reference_P.rowptr
    @test Array(level.P.colval) == reference_P.colval
    @test Array(level.P.nzval) ≈ reference_P.nzval
    finest = H.levels[1].A
    @test finest.nzval !== Adev.nzval
    @test H.pattern_rowptr === finest.rowptr
    @test H.pattern_colval === finest.colval
    old_cf = Array(H.levels[1].cf)

    @test resetup_amg!(H, Bdev, :memory) === H
    @test H.levels[1].A === finest
    @test Array(H.levels[1].cf) != old_cf
    @test H.workspace.stage_matrix1 === nothing
    @test H.workspace.stage_matrix2 === nothing
    check_pmis_galerkin(H)

    fresh = setup_amg(Bdev, opts)
    b = KAPreconditioners.backend_copy(backend, T.(1:n^2) ./ T(n^2))
    x = similar(b)
    KAPreconditioners.apply!(x, H, b)
    @test Array(x) ≈ Array(fresh * b)
    fill!(x, 0)
    rtol = T == Float32 ? 2.0e-6 : 1.0e-9
    solve!(x, H, b; rtol)
    @test norm(B * Array(x) - Array(b)) ≤ 2rtol * norm(b)

    first_level = H.levels[1]
    first_coarse = H.levels[2].A
    scaled = csr_matrix(T(2) * B; backend, use_vendor_linalg = false)
    @test resetup_amg!(H, scaled, :memory) === H
    @test H.levels[1].P === first_level.P
    @test H.levels[1].galerkin === first_level.galerkin
    @test H.levels[2].A === first_coarse
    @test Array(H * b) ≈ Array(fresh * b) / T(2)
    first_level = H.levels[1]
    @test resetup_amg!(H, Bdev, :partial_sparsity; n_levels_partial_keep = 1) === H
    @test H.levels[1] === first_level
    check_pmis_galerkin(H)

    perm = [2; 1; collect(3:n^2)]
    changed_graph = csr_matrix(B[perm, perm]; backend)
    @test nnz(changed_graph) == nnz(Bdev)
    @test_throws ArgumentError resetup_amg!(H, changed_graph, :memory)
    @test Array(H.levels[1].A.nzval) == Array(Bdev.nzval)
end

function check_pmis_rerank(backend)
    n = 64
    lo, hi = fill(-0.75, n - 1), fill(-1.0, n - 1)
    diagonal = fill(3.0, n)
    A = csr_matrix(spdiagm(-1 => lo, 0 => diagonal, 1 => hi); backend)
    B = csr_matrix(spdiagm(-1 => hi, 0 => diagonal, 1 => lo); backend)
    opts = AMGOptions(
        coarsening = PMIS(), interpolation = ClassicalInterpolation(0.0, 1),
        coarse_solver = :spai0, coarse_size = 4
    )
    H = setup_amg(A, opts)
    old = H.levels[1]
    initial_strength = Array(old.strength)
    initial_split = Array(old.cf)
    initial_cols = Array(old.P.colval)
    resetup_amg!(H, B, :memory)
    current = H.levels[1]
    @test Array(current.strength) == initial_strength
    @test Array(current.cf) == initial_split
    @test Array(current.P.colval) != initial_cols
    check_pmis_galerkin(H)
end

@testset "PMIS symbolic AMG setup on KA CPU" begin
    backend = KAPreconditioners.KernelAbstractions.CPU()
    check_pmis_memory(backend, Float64, ExtendedIInterpolation(0.1, 2, 2), :lu)
    check_pmis_memory(backend, Float32, ClassicalInterpolation(), :spai0)
    check_pmis_rerank(backend)

    # Directed strength with an isolated row exercises split repair and
    # empty interpolation rows. CSC input also exercises reset dispatch.
    A = sparse([4.0 -1 0 0; 0 4 -1 0; 0 0 4 0; 0 0 0 4])
    H = setup_amg(A, AMGOptions(coarsening = PMIS(), coarse_size = 1))
    @test H.levels[1].cf[4] == -3
    check_pmis_galerkin(H)
    @test resetup_amg!(H, 2A, :memory) === H
    @test H * ones(4) ≈ setup_amg(2A, H.options) * ones(4)
end

if Base.find_package("CUDA") !== nothing
    import CUDA
    if CUDA.functional()
        @testset "PMIS symbolic AMG setup on CUDA" begin
            CUDA.allowscalar(false)
            backend = CUDA.CUDABackend()
            check_pmis_memory(backend, Float32, ExtendedIInterpolation(0.1, 2, 2), :spai0)
            check_pmis_memory(backend, Float64, ExtendedIInterpolation(), :lu)
            check_pmis_rerank(backend)
        end
    end
end
