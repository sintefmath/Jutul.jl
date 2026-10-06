using Jutul, Jutul.KAPreconditioners, CUDA, LinearAlgebra, SparseArrays, Test

if CUDA.functional()
    @testset "CUDA fixed-layout hybrid GS/SSOR" begin
        backend = CUDA.CUDABackend()
        test_hybrid_gauss_seidel(backend)
        # A longer chain exercises the coefficient-only shared-memory plan
        # and the oversized-partition cuSPARSE fallback, with unchanged math.
        for n in (3000, 5000)
            A = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(4.0, n), 1 => fill(-0.5, n-1))
            config = HybridGaussSeidel(1, 0.7; omega = 0.8, partitions = 1)
            state = setup_smoother(csr_matrix(A; backend), config)
            native = state.native
            rhs = sin.(collect(1.0:n))
            b, x = CuArray(rhs), CUDA.zeros(Float64, n)
            KAPreconditioners.apply!(x, state, b)
            @test Array(x) ≈ reference_hybrid_gs(A, rhs, zeros(n), config, 1)
            B = A + spdiagm(0 => 0.1.*collect(1:n)./n)
            update_smoother!(state, csr_matrix(B; backend))
            @test state.native === native
            KAPreconditioners.apply!(x, state, b)
            @test Array(x) ≈ reference_hybrid_gs(B, rhs, zeros(n), config, 1)
        end
        # Multiple steps/W cycles and stream changes must work with retained
        # graph pointers after numerical reset.
        for cycle in (:V, :W)
            A = poisson_2d(8)
            H = setup_amg(csr_matrix(A; backend), AMGOptions(
                smoother = HybridGaussSeidel(2, 0.7; omega = 0.8, partitions = 4),
                aggressive_levels = 1, coarse_size = 4, cycle = cycle))
            b, x = CUDA.ones(Float64, size(A,1)), CUDA.zeros(Float64, size(A,1))
            KAPreconditioners.apply!(x, H, b)
            if !isnothing(H.execution)
                execution = H.execution
                expected = similar(x)
                KAPreconditioners.vcycle!(expected, b, H, 1; residual = b, zero_initial = true)
                @test Array(x) ≈ Array(expected)
                CUDA.stream!(CUDA.CuStream()) do
                    KAPreconditioners.apply!(x, H, b)
                    CUDA.synchronize()
                end
                @test H.execution === execution
                @test Array(x) ≈ Array(expected)
                resetup_amg!(H, csr_matrix(A+spdiagm(0 => fill(0.2,size(A,1))); backend), :operators)
                @test H.execution === execution
                KAPreconditioners.apply!(x, H, b)
                KAPreconditioners.vcycle!(expected, b, H, 1; residual = b, zero_initial = true)
                @test Array(x) ≈ Array(expected)
            end
        end
    end
end
