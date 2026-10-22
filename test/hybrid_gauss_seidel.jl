using Jutul, Test, LinearAlgebra, SparseArrays
using Jutul.KAPreconditioners

@testset "Hybrid Gauss-Seidel" begin
    A = sparse([4.0 -1 0 0; -1 4 -1 0; 0 -1 4 -1; 0 0 -1 4])
    b = collect(1.0:4.0)
    config = HybridGaussSeidel(; npartitions = 2, min_partition_size = 1)
    state = setup_smoother(A, config)
    # Local triangular solve, with the l1 shift on the two interface rows.
    M = LowerTriangular([4.0 0 0 0; -1 4.5 0 0; 0 0 4.5 0; 0 0 -1 4])
    initial = [0.1, 0.2, 0.3, 0.4]
    x = copy(initial)
    smooth!(x, state, state.matrix, b)
    @test x ≈ initial + M \ (b - A * initial)
    forward = M \ b
    @test state * b ≈ forward + M' \ (b - A * forward)

    H = setup_amg(csr_matrix(A), AMGOptions(; smoother = config, coarse_size = 1))
    x = zeros(4)
    solve!(x, H, b; rtol = 1.0e-8)
    @test A * x ≈ b
    resetup_amg!(H, csr_matrix(1.5A), :sparsity)
    fill!(x, 0)
    solve!(x, H, b; rtol = 1.0e-8)
    @test 1.5A * x ≈ b
end
