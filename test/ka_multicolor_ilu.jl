using Jutul, Jutul.KAPreconditioners, KernelAbstractions, StaticArrays
using SparseArrays, LinearAlgebra, Test, CUDA
const KP = Jutul.KAPreconditioners
BLAS.set_num_threads(1)

@testset "permuted ILU independent permutation reference" begin
    backends = CUDA.functional() ? (CPU(), CUDABackend()) : (CPU(),)
    for backend in backends, block in (false, true), factory in (ILU0, DILU), ordering in (:multicolor, :level)
        A = spdiagm(-1 => fill(-0.7, 30), 0 => fill(4.0, 31), 1 => fill(-1.1, 30))
        A[2, 15] = -0.2 # one-sided edge: graph must be symmetrized
        if block
            A = SparseMatrixCSC(
                size(A)..., copy(A.colptr), copy(A.rowval),
                [SMatrix{2, 2}(v, 0.1v, -0.07v, 1.3v) for v in nonzeros(A)]
            )
        end
        T = block ? SVector{2, Float64} : Float64
        b = block ? [T(sin(i), cos(i)) for i in 1:31] : sin.(1:31)
        C = csr_matrix(A; backend)
        source_values = Array(C.nzval)
        config = factory(; ordering)
        state = setup_smoother(C, config)
        p = Int.(Array(state.permutation))
        ref = setup_smoother(csr_matrix(A[p, p]), factory())
        expected = similar(b)
        expected[p] = ref \ b[p]
        rhs = KP.backend_copy(backend, b)
        x = similar(rhs)
        KP.apply!(x, state, rhs)
        @test Array(x) ≈ expected
        original_cache = state.apply_cache
        if ordering == :level
            natural_ref = setup_smoother(csr_matrix(A), factory())
            @test Array(x) ≈ natural_ref \ b
            inverse = invperm(p)
            ii, jj, _ = findnz(A)
            @test all((i < j) == (inverse[i] < inverse[j]) for (i, j) in zip(ii, jj))
            if backend isa CUDABackend
                @test !isnothing(original_cache)
            end
        end
        # The two colors must separate every directed edge.
        inner = state.inner
        @test state.matrix.nzval === (inner isa KP.ILU0State ? inner.factors : inner.values)
        @test Array(C.nzval) == source_values
        @test length(inner.factor_offsets) <= length(KP.ilu_symbolic(csr_matrix(A)).factor_offsets)
        # Application must support aliased input/output.
        copyto!(x, rhs)
        KP.apply!(x, state, x)
        @test Array(x) ≈ expected
        # Updates retain symbolic data and buffers, including equivalent new CSR storage.
        old_positions = state.value_positions
        D = csr_matrix(1.1A; backend)
        @test setup_smoother(D, config; reuse = state) === state
        @test state.value_positions === old_positions
        original_factor_cache = state.factor_cache
        if ordering == :level && backend isa CUDABackend
            @test !isnothing(original_factor_cache)
        end
        KP.apply!(x, state, rhs)
        @test Array(x) ≈ expected / 1.1
        @test state.apply_cache === original_cache
        # General smoothing and the AMG cycle hooks use residual correction.
        copyto!(x, rhs)
        expected_smooth = b + (ref \ (b[p] - (1.1A * b)[p]))[invperm(p)] / 1.1
        smooth!(x, state, D, rhs; steps = 1)
        @test Array(x) ≈ expected_smooth
        @test_throws ArgumentError update_smoother!(
            state, csr_matrix(
                A +
                    sparse([1], [31], [one(eltype(A))], 31, 31); backend
            )
        )
        # Exercise the native fused sweep kernels, including empty lower/upper
        # ranges in the first/last row, on both host and accelerator storage.
        natural = setup_smoother(C, factory())
        fused = copy(rhs)
        KP.ilu_smooth_result!(fused, C, rhs, natural)
        host_matrix = csr_matrix(A)
        host_state = setup_smoother(host_matrix, factory())
        host_fused = copy(b)
        KP.ilu_smooth_result!(host_fused, host_matrix, b, host_state)
        @test Array(fused) ≈ host_fused
        if ordering == :level
            damped = factory(1, 0.7; ordering)
            @test setup_smoother(D, damped; reuse = state) === state
            @test state.factor_cache === original_factor_cache
            damped_ref = setup_smoother(csr_matrix(1.1A[p, p]), factory(1, 0.7))
            expected[p] = damped_ref \ b[p]
            KP.apply!(x, state, rhs)
            @test Array(x) ≈ expected
            if backend isa CUDABackend
                @test state.apply_cache !== original_cache
            end
            # A different permutation strategy must rebuild its symbolic data.
            @test setup_smoother(D, factory(; ordering = :multicolor); reuse = state) !== state
        end
    end
end

@testset "permuted AMG and wrapper" begin
    A = spdiagm(-1 => fill(-1.0, 63), 0 => fill(3.0, 64), 1 => fill(-1.0, 63))
    for factory in (ILU0, DILU), ordering in (:multicolor, :level)
        config = factory(; ordering)
        H = setup_amg(A, AMGOptions(; smoother = config, coarse_size = 8))
        b = sin.(1:64)
        x = zeros(64)
        KP.apply!(x, H, b)
        @test norm(b - A * x) < norm(b)
        resetup_amg!(H, 1.1A, :operators)
        KP.apply!(x, H, b)
        @test norm(b - 1.1A * x) < norm(b)
        precond = Jutul.KASmootherPreconditioner(config)
        Jutul.update_preconditioner!(precond, csr_matrix(A), b, nothing, nothing)
        Jutul.apply!(x, precond, b)
        @test x ≈ precond.factor \ b
    end
    @test_throws ArgumentError ILU0(; ordering = :invalid)
    @test_throws ArgumentError DILU(; ordering = :invalid)
end

include("ka_ilu_graph_replay.jl")
include("ka_ilu_auto_ordering.jl")
