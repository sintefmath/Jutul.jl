using Jutul
using CUDA
using LinearAlgebra
using SparseArrays
using Test

if CUDA.functional()
    @testset "CUDA cuSOLVERRf Schur block" begin
        T = spdiagm(-1 => fill(-1.0, 4), 0 => fill(4.0, 5), 1 => fill(-1.0, 4))
        E = spdiagm(-1 => fill(-1.0, 4), 1 => fill(-1.0, 4))
        A = kron(sparse(I, 5, 5), T) + kron(E, sparse(I, 5, 5))
        b = collect(1.0:size(A, 1))
        backend = CUDA.CUDABackend()
        matrix = Jutul.KAPreconditioners.csr_matrix(A; backend)
        F = Jutul.KernelExecution.factorize_linear_system(lu, matrix)
        @test F isa Jutul.KAPreconditioners.SparseLU
        @test F.factorization isa Base.get_extension(Jutul, :JutulCUDAExt).CUDARFFactor
        rhs = CuArray(b)
        x = similar(rhs)
        ldiv!(x, F, rhs)
        @test Array(x) ≈ A \ b

        updated = Jutul.KAPreconditioners.csr_matrix(1.5A; backend)
        @test Jutul.KernelExecution.refactorize_linear_system!(
            lu!, F, updated
        ) === F
        ldiv!(x, F, rhs)
        @test Array(x) ≈ (1.5A) \ b

        different = copy(A)
        different[1, 3] = 0.1
        @test_throws ArgumentError Jutul.KernelExecution.refactorize_linear_system!(
            lu!, F, Jutul.KAPreconditioners.csr_matrix(different; backend)
        )
    end

    @testset "CUDA sparse AMG coarse solve" begin
        T = spdiagm(-1 => fill(-1.0, 4), 0 => fill(4.0, 5), 1 => fill(-1.0, 4))
        E = spdiagm(-1 => fill(-1.0, 4), 1 => fill(-1.0, 4))
        A = kron(sparse(I, 5, 5), T) + kron(E, sparse(I, 5, 5))
        backend = CUDA.CUDABackend()
        matrix = Jutul.KAPreconditioners.csr_matrix(A; backend)
        H = Jutul.KAPreconditioners.setup_amg(
            matrix, Jutul.KAPreconditioners.AMGOptions(coarse_size = 8)
        )
        coarse = H.levels[end].coarse_solver
        @test coarse isa Jutul.KAPreconditioners.SparseLU
        @test coarse.factorization isa Base.get_extension(Jutul, :JutulCUDAExt).CUDARFFactor
        b = CUDA.ones(Float64, size(A, 1))
        x = CUDA.zeros(Float64, size(A, 1))
        Jutul.KAPreconditioners.cycle!(x, H, b)
        @test norm(ones(size(A, 1)) - A * Array(x)) < norm(ones(size(A, 1)))

        changed = Jutul.KAPreconditioners.csr_matrix(1.2A; backend)
        Jutul.KAPreconditioners.resetup_amg!(H, changed, :operators)
        @test H.levels[end].coarse_solver === coarse
        fill!(x, 0.0)
        Jutul.KAPreconditioners.cycle!(x, H, b)
        @test norm(ones(size(A, 1)) - (1.2A) * Array(x)) < norm(ones(size(A, 1)))
    end

    @testset "CUDA sparse LU repivots on resetup" begin
        A = sparse([10.0 1.0; 1.0 10.0])
        backend = CUDA.CUDABackend()
        F = Jutul.KAPreconditioners.setup_sparse_lu(
            Jutul.KAPreconditioners.csr_matrix(A; backend)
        )
        initial_factor = F.factorization
        p = Array(initial_factor.p) .+ 1
        q = Array(initial_factor.q) .+ 1
        changed = copy(A)
        changed[p[1], q[1]] = 0.0
        @test nnz(changed) == nnz(A)
        @test !iszero(det(Matrix(changed)))

        matrix = Jutul.KAPreconditioners.csr_matrix(changed; backend)
        @test Jutul.KAPreconditioners.resetup_sparse_lu!(F, matrix) === F
        @test F.factorization !== initial_factor
        b = CuArray([1.0, 2.0])
        x = similar(b)
        ldiv!(x, F, b)
        @test Array(x) ≈ changed \ Array(b)

        @test Jutul.KAPreconditioners.resetup_sparse_lu!(
            F, Jutul.KAPreconditioners.csr_matrix(A; backend)
        ) === F
        ldiv!(x, F, b)
        @test Array(x) ≈ A \ Array(b)
    end
end
