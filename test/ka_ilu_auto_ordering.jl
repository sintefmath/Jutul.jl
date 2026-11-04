@testset "automatic ILU wrapper ordering and graph reuse" begin
    backends = CUDA.functional() ? (CPU(), CUDABackend()) : (CPU(),)
    for backend in backends, (method, factory) in ((:ilu0, ILU0), (:dilu, DILU))
        B = SMatrix{2, 2, Float32}(4, 0.1, -0.2, 3)
        A = spdiagm(-1 => fill(-0.1f0B, 15), 0 => fill(B, 16), 1 => fill(-0.2f0B, 15))
        A[2, 13] = -0.03f0B
        C = csr_matrix(A; backend)
        rhs = KP.backend_copy(backend, Float32.(sin.(1:32)))
        x = similar(rhs)
        reference = similar(rhs)
        automatic = Jutul.KASmootherPreconditioner(method)
        natural = Jutul.KASmootherPreconditioner(method; ordering = :natural)
        @test automatic.config.ordering == :auto
        factor = nothing
        factor_cache = nothing
        apply_cache = nothing
        for change in 1:3
            D = copy(A)
            for i in 1:16
                D[i, i] += (0.01f0i + 0.17f0change) * one(B)
            end
            C = csr_matrix(D; backend)
            Jutul.update_preconditioner!(automatic, C, rhs, nothing, nothing)
            Jutul.update_preconditioner!(natural, C, rhs, nothing, nothing)
            Jutul.apply!(x, automatic, rhs)
            Jutul.apply!(reference, natural, rhs)
            @test Array(x) ≈ Array(reference) rtol = 1.0e-5
            @test natural.factor isa Union{KP.ILU0State, KP.DILUState}
            @test automatic.factor.config.ordering == (backend isa CUDABackend ? :level : :natural)
            if change == 1
                factor = automatic.factor
            else
                @test automatic.factor === factor
            end
            if backend isa CUDABackend
                @test automatic.factor isa KP.MulticolorILUState
                if change == 1
                    apply_cache = automatic.factor.apply_cache
                    @test !isnothing(apply_cache)
                else
                    @test automatic.factor.apply_cache === apply_cache
                    if change == 2
                        factor_cache = automatic.factor.factor_cache
                        @test !isnothing(factor_cache)
                    else
                        @test automatic.factor.factor_cache === factor_cache
                    end
                end
            end
        end
        wrong_precision = KP.backend_copy(backend, Float64.(1:32))
        @test_throws ArgumentError Jutul.apply!(wrong_precision, automatic, wrong_precision)
        @test_throws ArgumentError Jutul.apply!(x, automatic, wrong_precision)
        # Direct configs can request the same backend selection, including AMG resetup.
        config = factory(; ordering = :auto)
        H = setup_amg(
            csr_matrix(
                spdiagm(
                    -1 => fill(-1.0f0, 31), 0 => fill(4.0f0, 32),
                    1 => fill(-1.0f0, 31)
                ); backend
            ), AMGOptions(; smoother = config, coarse_size = 8)
        )
        b = KP.backend_copy(backend, ones(Float32, 32))
        KP.apply!(x, H, b)
        resetup_amg!(
            H, csr_matrix(
                spdiagm(
                    -1 => fill(-1.0f0, 31), 0 => fill(5.0f0, 32),
                    1 => fill(-1.0f0, 31)
                ); backend
            ), :operators
        )
        KP.apply!(x, H, b)
        @test all(isfinite, Array(x))
        @test H.levels[1].smoother.config.ordering == (backend isa CUDABackend ? :level : :natural)
    end
    @test Jutul.KASmootherPreconditioner(:spai0).config isa SPAI0
    @test Jutul.KASmootherPreconditioner(:vendor_ilu).config isa VendorILU
end
