using Jutul, CUDA, KernelAbstractions, LinearAlgebra, SparseArrays, StaticArrays, Test
import Adapt

function test_vendor_ilu_reuse(backend)
    for T in (Float32, Float64), block in (false, true)
        n = 4
        diagonal = block ? SMatrix{3, 3, T}(4, 0.1, 0, 0.2, 5, 0.1, 0, 0.2, 6) : T(4)
        off_diagonal = block ? -SMatrix{3, 3, T}(I) : -one(T)
        vector_unit = block ? SVector{3, T}(1, 1, 1) : one(T)
        host = sparse(vcat(1:n, 2:n, 1:n-1), vcat(1:n, 1:n-1, 2:n),
            vcat(fill(diagonal, n), fill(off_diagonal, 2n-2)), n, n)
        matrix = Jutul.KAPreconditioners.csr_matrix(host; backend)
        state = Jutul.KAPreconditioners.setup_smoother(matrix, Jutul.KAPreconditioners.VendorILU())
        factor = state.factor
        factor_values = state.factor_values
        rhs = fill(vector_unit, n)
        allocation = Adapt.adapt(backend, fill(-7vector_unit, n + 2))
        output = Jutul.KAPreconditioners.buffer_prefix(allocation, n)
        device_rhs = Adapt.adapt(backend, rhs)
        for scale in (one(T), T(1.5), T(0.8))
            matrix.nzval .= scale .* Adapt.adapt(backend, nonzeros(host))
            @test Jutul.KAPreconditioners.setup_smoother(matrix,
                Jutul.KAPreconditioners.VendorILU(); reuse = state) === state
            Jutul.KAPreconditioners.apply!(output, state, device_rhs)
            @test (scale .* host) * Array(output) ≈ rhs
            @test Array(allocation)[n+1:end] == fill(-7vector_unit, 2)
            @test state.factor === factor
            @test state.factor_values === factor_values
        end
    end
end

if CUDA.functional()
    @testset "CUDA vendor ILU factor and vector reuse" begin
        test_vendor_ilu_reuse(CUDA.CUDABackend())
    end
end

if isdefined(Main, :AMDGPU) && AMDGPU.functional()
    @testset "AMDGPU vendor ILU factor and vector reuse" begin
        test_vendor_ilu_reuse(AMDGPU.ROCBackend())
    end
end
