using CUDA
using Jutul
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using Test

@testset "KA block sparse LU" begin
    backends = CUDA.functional() ? (CPU(), CUDA.CUDABackend()) : (CPU(),)
    for backend in backends
        @testset "$(typeof(backend))" begin
            # Three independent components, with rows interleaved in the CSR.
            A = sparse(
                Float32[
                    10 0 2 0 0 0
                    0 5 0 1 0 0
                    3 0 4 0 0 0
                    0 -1 0 4 0 2
                    0 0 0 0 6 0
                    0 0 0 0 0 7
                ]
            )
            function csr(A)
                raw = Jutul.KAPreconditioners.csr_matrix(A; backend, index_type = Int32)
                return Jutul.KAPreconditioners.csr_matrix(
                    raw.rowptr, raw.colval, raw.nzval, size(raw)...; backend
                )
            end
            matrix = csr(A)
            F = Jutul.KASparseFactorizations.try_factorize(matrix)
            @test F isa Jutul.KASparseFactorizations.BlockSparseLU
            if !applicable(Jutul.KAPreconditioners.setup_preferred_sparse_lu, matrix)
                @test Jutul.KernelExecution.factorize_linear_system(lu, matrix) isa typeof(F)
            end
            @test length(F.factors) == 2^2 + 3^2 + 1
            b = Jutul.KAPreconditioners.backend_copy(backend, Float32[1, 2, 3, 4, 5, 6])
            x = similar(b)
            ldiv!(x, F, b)
            @test A * Array(x) ≈ Array(b)

            changed = copy(A)
            changed[1, 3] = 0
            dropzeros!(changed)
            changed[1, 2] = 1
            @test_throws ArgumentError Jutul.KASparseFactorizations.refactorize!(F, csr(changed))

            # Changing only values forces a different pivot in the first block.
            A[1, 1] = 0
            @test Jutul.KernelExecution.refactorize_linear_system!(lu!, F, csr(A)) === F
            copyto!(x, b)
            ldiv!(x, F, x)
            @test A * Array(x) ≈ Array(b)

            singular = copy(A)
            singular[5, 5] = 0
            @test_throws SingularException Jutul.KASparseFactorizations.refactorize!(F, csr(singular))
            Jutul.KASparseFactorizations.refactorize!(F, csr(A))
            ldiv!(x, F, b)
            @test A * Array(x) ≈ Array(b)

            connected = sparse(Float32[4 1; 1 4])
            @test isnothing(Jutul.KASparseFactorizations.try_factorize(csr(connected)))
        end
    end
end
