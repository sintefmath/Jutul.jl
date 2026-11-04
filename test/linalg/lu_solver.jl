using Jutul, Test, LinearAlgebra, SparseArrays
using KLU

function lu_solver_test_matrix(n)
    # Nonsymmetric, diagonally dominant matrix with a fixed sparsity pattern
    A = spdiagm(0 => fill(4.0, n), 1 => fill(-1.0, n - 1), -1 => fill(-0.5, n - 1))
    A[1, n] = 0.25
    return A
end

@testset "Sparse direct solvers" begin
    n = 12
    A = lu_solver_test_matrix(n)
    rhs = collect(1.0:n)
    tol = 1.0e-12 * norm(rhs)
    for (solver_type, factorization_type) in ((LUSolver, SparseArrays.UMFPACK.UmfpackLU), (KLUSolver, KLU.KLUFactorization))
        @testset "$solver_type" begin
            solver = solver_type()
            system = Jutul.LinearizedSystem(A, copy(rhs))
            result = Jutul.linear_solve!(system, solver)
            @test result.ok
            @test solver.F isa factorization_type
            @test norm(A * system.dx + rhs) <= tol
            # New values on the same pattern refactorize the stored object in place
            F = solver.F
            A2 = 2.0 * A
            system2 = Jutul.LinearizedSystem(A2, copy(rhs))
            Jutul.linear_solve!(system2, solver)
            @test solver.F === F
            @test norm(A2 * system2.dx + rhs) <= tol

            @testset "No factorization reuse" begin
                solver = solver_type(reuse_memory = false)
                for matrix in (A, A2)
                    system = Jutul.LinearizedSystem(matrix, copy(rhs))
                    result = Jutul.linear_solve!(system, solver)
                    @test result.ok
                    @test solver.F === nothing
                    @test norm(matrix * system.dx + rhs) <= tol
                end
            end
        end
    end
    @testset "KLU explicit buffers" begin
        system = Jutul.LinearizedSystem(A, zeros(n))
        fill!(system.dx, NaN)
        dx = similar(rhs)
        r = copy(rhs)
        result = Jutul.linear_solve!(system, KLUSolver(max_size = n); dx = dx, r = r)
        @test result.ok
        @test norm(A * dx + rhs) <= tol
        @test r == rhs
        @test all(isnan, system.dx)
        @test all(iszero, system.r)
    end
    @testset "KLU safeguards" begin
        system = Jutul.LinearizedSystem(A, copy(rhs))
        @test_throws ErrorException Jutul.linear_solve!(system, KLUSolver(max_size = n - 1))
        dense_system = (jac = Matrix(A), dx = zeros(n), r = copy(rhs))
        @test_throws ArgumentError Jutul.linear_solve!(dense_system, KLUSolver())

        nonfinite_rhs = copy(rhs)
        nonfinite_rhs[1] = NaN
        system = Jutul.LinearizedSystem(A, nonfinite_rhs)
        @test_throws ErrorException Jutul.linear_solve!(system, KLUSolver())
        result = Jutul.linear_solve!(system, KLUSolver(check = false))
        @test result.ok
        @test !all(isfinite, system.dx)
    end
    @testset "Simulation with KLU" begin
        g = CartesianMesh((4, 4), (1.0, 1.0))
        model = SimulationModel(DiscretizedDomain(g), SimpleHeatSystem())
        nc = number_of_cells(g)
        T0 = [1.0 + sin(0.3 * i) for i in 1:nc]
        lu_states, klu_states = map((LUSolver, KLUSolver)) do solver_type
            sim = Simulator(model, state0 = setup_state(model, Dict(:T => T0)))
            result, = simulate(sim, [1.0]; info_level = -1, linear_solver = solver_type())
            return result[end][:T]
        end
        @test isapprox(klu_states, lu_states, rtol = 1.0e-10, atol = 1.0e-12)
    end
end
