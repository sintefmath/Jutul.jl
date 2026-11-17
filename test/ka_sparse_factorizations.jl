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

            @testset "Many independent well blocks" begin
                nblocks, blocksize = 2049, 8
                block = spdiagm(
                    -1 => fill(-1.0f0, blocksize - 1), 0 => fill(4.0f0, blocksize),
                    1 => fill(-1.0f0, blocksize - 1)
                )
                A = kron(sparse(I, nblocks, nblocks), block)
                matrix = csr(A)
                F = Jutul.KASparseFactorizations.try_factorize(matrix)
                @test F isa Jutul.KASparseFactorizations.BlockSparseLU
                @test length(F.factors) == nblocks * blocksize^2
                b = Jutul.KAPreconditioners.backend_copy(backend, ones(Float32, size(A, 1)))
                x = similar(b)
                nonzeros(matrix) .*= 1.25f0
                @test Jutul.KernelExecution.refactorize_linear_system!(lu!, F, matrix) === F
                ldiv!(x, F, b)
                @test 1.25f0 .* (A * Array(x)) ≈ Array(b)
            end
        end
    end
end

@testset "LUSolver on KA CSR systems" begin
    CUDA.functional() && CUDA.allowscalar(false)
    backends = CUDA.functional() ? (CPU(), CUDA.CUDABackend()) : (CPU(),)
    for backend in backends, T in (Float32, Float64), disconnected in (false, true)
        @testset "$(typeof(backend)), $T, disconnected=$disconnected" begin
            A = sparse(T[4 1 0; 2 5 1; 0 -1 3])
            if disconnected
                A = blockdiag(A, sparse(T[3 1; -1 4]))
            end
            raw = Jutul.KAPreconditioners.csr_matrix(A; backend)
            matrix = Jutul.KAPreconditioners.csr_matrix(
                raw.rowptr, raw.colval, raw.nzval, size(raw)...; backend)
            rhs = T.(1:size(A, 1))
            r = Jutul.KAPreconditioners.backend_copy(backend, rhs)
            dx = similar(r)
            system = Jutul.LinearizedSystem(
                matrix, r, dx, nonzeros(matrix), r, dx, Jutul.EquationMajorLayout())
            solver = LUSolver()
            result = Jutul.linear_solve!(system, solver)
            @test result.ok
            @test A*Array(dx) ≈ -rhs
            @test Array(r) == rhs
            factorization = solver.F
            nonzeros(matrix) .*= T(1.25)
            Jutul.linear_solve!(system, solver)
            @test solver.F === factorization
            @test T(1.25)*(A*Array(dx)) ≈ -rhs

            # Explicit output/rhs buffers must not read or overwrite the
            # system's default buffers, including its non-finite output.
            fill!(dx, T(NaN))
            output = similar(dx)
            alternate_rhs = Jutul.KAPreconditioners.backend_copy(backend, reverse(rhs))
            result = Jutul.linear_solve!(system, solver; dx=output, r=alternate_rhs)
            @test result.ok
            @test T(1.25)*(A*Array(output)) ≈ -reverse(rhs)
            @test all(isnan, Array(dx))
            @test Array(alternate_rhs) == reverse(rhs)
            @test Array(r) == rhs

            solver = LUSolver(reuse_memory=false)
            Jutul.linear_solve!(system, solver)
            @test isnothing(solver.F)
            @test T(1.25)*(A*Array(dx)) ≈ -rhs
            @test_throws ErrorException Jutul.linear_solve!(
                system, LUSolver(max_size=length(rhs)-1))
        end
    end
end
