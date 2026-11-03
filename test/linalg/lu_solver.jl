using Jutul, Test, LinearAlgebra, SparseArrays
using KLU

function lu_solver_test_matrix(n)
    # Nonsymmetric, diagonally dominant matrix with a fixed sparsity pattern
    A = spdiagm(0 => fill(4.0, n), 1 => fill(-1.0, n - 1), -1 => fill(-0.5, n - 1))
    A[1, n] = 0.25
    return A
end

@testset "LUSolver backends" begin
    n = 12
    A = lu_solver_test_matrix(n)
    rhs = collect(1.0:n)
    tol = 1.0e-12 * norm(rhs)
    for backend in (:umfpack, :klu, :auto)
        @testset "$backend" begin
            solver = LUSolver(backend = backend)
            system = Jutul.LinearizedSystem(A, copy(rhs))
            result = Jutul.linear_solve!(system, solver)
            @test result.ok
            @test norm(A * system.dx + rhs) <= tol
            # New values on the same pattern refactorize the stored object in place
            F = solver.F
            A2 = 2.0 * A
            system2 = Jutul.LinearizedSystem(A2, copy(rhs))
            Jutul.linear_solve!(system2, solver)
            @test solver.F === F
            @test norm(A2 * system2.dx + rhs) <= tol
        end
    end
    @testset "Backend selection" begin
        @test LUSolver().backend == :umfpack

        solver = LUSolver(backend = :auto, klu_max_size = n - 1)
        Jutul.linear_solve!(Jutul.LinearizedSystem(A, copy(rhs)), solver)
        @test solver.backend == :umfpack
        @test solver.F isa SparseArrays.UMFPACK.UmfpackLU

        solver = LUSolver(backend = :auto, klu_max_size = n)
        Jutul.linear_solve!(Jutul.LinearizedSystem(A, copy(rhs)), solver)
        @test solver.backend == :klu
        @test solver.F isa KLU.KLUFactorization

        # A fill limit below the actual fill makes :auto fall back to UMFPACK
        solver = LUSolver(backend = :auto, klu_max_fill = 0.0)
        Jutul.linear_solve!(Jutul.LinearizedSystem(A, copy(rhs)), solver)
        @test solver.backend == :umfpack
        @test solver.F isa SparseArrays.UMFPACK.UmfpackLU

        # Changing the backend of a solver with a stored factorization refactorizes
        solver.backend = :klu
        system = Jutul.LinearizedSystem(A, copy(rhs))
        Jutul.linear_solve!(system, solver)
        @test solver.F isa KLU.KLUFactorization
        @test norm(A * system.dx + rhs) <= tol

        @test_throws ArgumentError LUSolver(backend = :cholesky)
    end
    @testset "Simulation with KLU" begin
        g = CartesianMesh((4, 4), (1.0, 1.0))
        model = SimulationModel(DiscretizedDomain(g), SimpleHeatSystem())
        nc = number_of_cells(g)
        T0 = [1.0 + sin(0.3 * i) for i in 1:nc]
        states = Dict{Symbol, Any}()
        for backend in (:umfpack, :klu)
            sim = Simulator(model, state0 = setup_state(model, Dict(:T => T0)))
            result, = simulate(sim, [1.0]; info_level = -1, linear_solver = LUSolver(backend = backend))
            states[backend] = result[end][:T]
        end
        @test isapprox(states[:klu], states[:umfpack], rtol = 1.0e-10)
    end
end
