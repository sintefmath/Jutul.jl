using Jutul, SparseArrays, Test

@testset "Independent Float64 GS diagonal updates" begin
    KP = Jutul.KAPreconditioners
    n = 16_391
    function compare_serial!(state, A)
        invoke(KP.update_smoother!, Tuple{KP.GaussSeidelState, typeof(A)}, state, A)
        expected = copy(state.inverse_diagonal)
        KP.update_smoother!(state, A)
        @test isequal(state.inverse_diagonal, expected)
    end
    for Ti in (Int32, Int64), value in (1.0, -1.0, Inf, NaN, 1e-320)
        A = KP.csr_matrix(spdiagm(0 => fill(value, n)); index_type = Ti)
        state = KP.setup_smoother(A, KP.GaussSeidel())
        compare_serial!(state, A)
    end
    for Ti in (Int32, Int64)
        A = Jutul.StaticSparsityMatrixCSR(repeat([1.0, -0.25], n),
            repeat(Ti.(1:n); inner = 2), Ti.(1:2:2n+1), n, n,
            KP.KernelAbstractions.CPU())
        state = KP.setup_smoother(A, KP.GaussSeidel())
        compare_serial!(state, A)
    end

    A = KP.csr_matrix(spdiagm(0 => ones(n)); index_type = Int32)
    state = KP.setup_smoother(A, KP.GaussSeidel())
    A.nzval[17] = -0.0
    A.nzval[9] = 0.0
    failure = try
        KP.update_smoother!(state, A)
        nothing
    catch e
        e
    end
    @test failure isa ArgumentError
    @test occursin("row 9", sprint(showerror, failure))
    A.nzval[9], A.nzval[17] = 2.0, 3.0
    compare_serial!(state, A)
end
