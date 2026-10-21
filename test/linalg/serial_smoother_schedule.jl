using Jutul, Test, SparseArrays, LinearAlgebra
using Jutul.StaticArrays: SMatrix, SVector

@testset "Natural and level ILU traversals agree" begin
    kp = Jutul.KAPreconditioners
    n, width = 96, 16
    scalar = spdiagm(-width => fill(-0.3, n-width), 0 => fill(5.0, n),
        1 => fill(-0.1, n-1), width => fill(-0.2, n-width))
    indices_i, indices_j, coefficients = findnz(scalar)
    Block = SMatrix{3, 3, Float64, 9}
    blocks = map(indices_i, indices_j, coefficients) do i, j, value
        value * (i == j ? one(Block) : Block(1, .02, .03, .04, 1, .05, .06, .07, 1))
    end
    block = sparse(indices_i, indices_j, blocks, n, n)
    for M in (scalar, block), Ti in (Int32, Int64), config in (kp.ILU0(1, 0.7), kp.DILU(1, 0.7))
        A = kp.csr_matrix(M; index_type = Ti)
        natural = kp.setup_smoother(A, config)
        levels = deepcopy(natural)
        natural_order = (lower_offsets = (1, n+1), lower_rows = 1:n,
            upper_offsets = (1, n+1), upper_rows = n:-1:1)
        level_order = (lower_offsets = levels.factor_offsets, lower_rows = levels.factor_rows,
            upper_offsets = levels.upper_offsets, upper_rows = levels.upper_rows)
        # Refactor both from the same matrix, including asymmetric entries.
        for state in (natural, levels)
            copyto!(state isa kp.ILU0State ? state.factors : state.values, A.nzval)
        end
        Jutul.Polyester.disable_polyester_threads() do
            kp.ilu_factor_cpu!(natural, natural_order.lower_offsets, natural_order.lower_rows)
        end
        kp.ilu_factor_cpu!(levels, level_order.lower_offsets, level_order.lower_rows)
        @test natural.inverse_diagonal ≈ levels.inverse_diagonal rtol=1e-14
        if natural isa kp.ILU0State
            @test natural.factors ≈ levels.factors rtol=1e-14
        end
        rhs = M === scalar ? collect(range(0.1, 1.0; length=n)) :
            [SVector(sin(i), cos(i), 0.1i) for i in 1:n]
        kp.ensure_smoother_work!(natural, rhs)
        kp.ensure_smoother_work!(levels, rhs)
        expected, actual = similar(rhs), similar(rhs)
        kp.ilu_solve_cpu!(expected, levels, rhs, levels.work, level_order)
        Jutul.Polyester.disable_polyester_threads() do
            kp.ilu_solve_cpu!(actual, natural, rhs, natural.work, natural_order)
        end
        @test actual ≈ expected rtol=1e-14
        kp.ilu_solve_cpu!(actual, natural, rhs, natural.work)
        @test actual ≈ expected rtol=1e-14
        # Fused residual/correction uses both triangular traversals too.
        expected .= rhs
        actual .= rhs
        kp.ilu_smooth_result_cpu!(expected, A, rhs, levels, levels.work, levels.residual, level_order)
        Jutul.Polyester.disable_polyester_threads() do
            kp.ilu_smooth_result_cpu!(actual, A, rhs, natural, natural.work, natural.residual, natural_order)
        end
        @test actual ≈ expected rtol=1e-14
        actual .= rhs
        kp.ilu_smooth_result_cpu!(actual, A, rhs, natural, natural.work, natural.residual)
        @test actual ≈ expected rtol=1e-14
        schedule = kp.cpu_smoother_schedule(natural)
        if Threads.nthreads(:default) == 1
            @test schedule.lower_rows == 1:n
            @test schedule.upper_rows == n:-1:1
            @test length(schedule.lower_offsets) == 2
        else
            @test schedule.lower_rows === natural.factor_rows
            @test schedule.upper_rows === natural.upper_rows
        end
    end
end
