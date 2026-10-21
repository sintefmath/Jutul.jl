using Jutul, Test, SparseArrays, LinearAlgebra

@testset "Changed CSR contents invalidate ILU symbolic reuse" begin
    kp = Jutul.KAPreconditioners
    for T in (Float32, Float64), Ti in (Int32, Int64), config in (kp.ILU0(), kp.DILU())
        M = sparse(T[4 -1 0 0; -1 4 0 0; 0 0 4 -1; 0 0 -1 4])
        changed = sparse(T[4 0 -1 0; 0 4 -1 0; -1 -1 4 0; 0 0 0 4])
        A = kp.csr_matrix(M; index_type = Ti)
        B = kp.csr_matrix(changed; index_type = Ti)
        state = kp.setup_smoother(A, config)
        @test kp.setup_smoother(A, config; reuse = state) === state
        # AMG owned_on_backend_reusing replaces structural contents before
        # requesting smoother reuse, while retaining these array objects.
        copyto!(A.rowptr, B.rowptr)
        copyto!(A.colval, B.colval)
        copyto!(A.nzval, B.nzval)
        reused = kp.setup_smoother(A, config; reuse = state)
        fresh = kp.setup_smoother(B, config)
        @test reused !== state
        @test reused.host_rowptr == B.rowptr
        @test reused.host_colval == B.colval
        rhs = T[1, 2, 3, 4]
        actual, expected = zeros(T, 4), zeros(T, 4)
        kp.apply!(actual, reused, rhs)
        kp.apply!(expected, fresh, rhs)
        @test actual ≈ expected
    end
end
