@testset "ILU graph replay with changing coefficients" begin
    backends = CUDA.functional() ? (CPU(), CUDABackend()) : (CPU(),)
    for backend in backends, T in (Float32, Float64), block in (false, true), factory in (ILU0, DILU)
        n = 16
        ii, jj, vv = Int[], Int[], T[]
        for i in 1:n
            push!(ii, i);push!(jj, i);push!(vv, T(4))
            for j in (i - 4, i - 1, i + 1, i + 4)
                1 <= j <= n || continue
                abs(j - i) == 1 && (i - 1) ÷ 4 != (j - 1) ÷ 4 && continue
                push!(ii, i);push!(jj, j);push!(vv, T(-0.7))
            end
        end
        A = sparse(ii, jj, vv, n, n)
        A[2, 15] = T(-0.2)
        if block
            values = [
                i == j ? SMatrix{2, 2, T}(v, T(0.1), T(-0.2), T(1.4) * v) :
                    SMatrix{2, 2, T}(v, T(0.03) * v, T(-0.12) * v, T(0.8) * v)
                    for (i, j, v) in zip(findnz(A)...)
            ]
            A = sparse(findnz(A)[1], findnz(A)[2], values, n, n)
        end
        b = block ? [SVector{2, T}(sin(T(i)), cos(T(i))) for i in 1:n] : sin.(T.(1:n))
        config = factory(1, 0.73; ordering = :level)
        state = setup_smoother(csr_matrix(A; backend), config)
        rhs = KP.backend_copy(backend, b)
        x = similar(rhs)
        factor_cache = nothing
        apply_cache = nothing
        for change in 1:3
            D = copy(A)
            for i in 1:n
                D[i, i] += T(0.17change + 0.01i) * one(eltype(A))
            end
            update_smoother!(state, csr_matrix(D; backend))
            KP.apply!(x, state, rhs)
            reference = setup_smoother(csr_matrix(D), factory(1, 0.73)) \ b
            @test Array(x) ≈ reference rtol = (T === Float32 ? 1.0e-5 : 1.0e-12)
            if backend isa CUDABackend
                if change == 1
                    factor_cache = state.factor_cache
                    apply_cache = state.apply_cache
                    @test !isnothing(factor_cache)
                    @test !isnothing(apply_cache)
                else
                    @test state.factor_cache === factor_cache
                    @test state.apply_cache === apply_cache
                end
            end
        end
    end
end
