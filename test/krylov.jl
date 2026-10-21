using Jutul, Test, LinearAlgebra, SparseArrays

@testset "Krylov workspace solves and reuse" begin
    for T in (Float32, Float64), method in (:gmres, :bicgstab, :fgmres, :cg)
        @testset "$T / $method" begin
            A = sparse(T[4.2 0.3 0 0; 0.3 5.1 0.7 0; 0 0.7 6.4 0.2; 0 0 0.2 7.3])
            context = DefaultContext()
            model = SimulationModel(ScalarTestDomain(), ScalarTestSystem(); context)
            tolerance = T == Float32 ? 2.0e-5 : 1.0e-11
            solver = GenericKrylov(
                method; relative_tolerance = tolerance / 10,
                absolute_tolerance = zero(T), precond_side = method == :cg ? :left : :right
            )
            residual = zeros(T, 4)
            increment = zeros(T, 4)
            system = Jutul.LinearizedSystem(A, residual, increment, nonzeros(A),
                residual, increment, Jutul.EquationMajorLayout())
            workspace = nothing
            for rhs in (T[1, 2, 3, 4], T[-2, 3, 1, 0.5], zeros(T, 4))
                system.r_buffer .= rhs
                result = Jutul.linear_solve!(system, solver, context, model,
                    nothing, 1.0, nothing)
                @test result.ok
                @test all(isfinite, system.dx_buffer)
                @test norm(A*system.dx_buffer + rhs) <= tolerance*max(norm(rhs), one(T))
                @test eltype(solver.storage.x) == T
                if isnothing(workspace)
                    workspace = solver.storage
                else
                    @test solver.storage === workspace
                end
            end
        end
    end
end

@testset "Krylov minimum-iteration callback" begin
    A = spdiagm(0 => collect(1.0:8.0))
    context = DefaultContext()
    model = SimulationModel(ScalarTestDomain(), ScalarTestSystem(); context)
    system = Jutul.LinearizedSystem(A, ones(8))
    solver = GenericKrylov(:gmres; min_iterations = 2, relative_tolerance = 1e-10)
    result = Jutul.linear_solve!(system, solver, context, model, nothing, 1.0, nothing)
    @test result.iterations >= 2
    @test norm(A*system.dx_buffer + ones(8)) < 1e-9
end

# Also run this block when the optional distributed extension has been loaded.
let extension = Base.get_extension(Jutul, :JutulPartitionedArraysExt)
    if !isnothing(extension)
        @testset "Distributed Krylov workspace constructors" begin
            PA = extension.PartitionedArrays
            parts = PA.DebugArray(LinearIndices((2,)))
            partition = PA.uniform_partition(parts, 8)
            rhs = PA.pvector(indices -> ones(length(indices)), partition)
            diagonal = PA.pvector(indices -> Float64.(PA.local_to_global(indices)) .+ 1.0, partition)
            for make_workspace in (extension.local_bicgstab_solver, extension.local_gmres_solver)
                workspace = make_workspace(rhs)
                @test typeof(workspace.x) == typeof(rhs)
                @test length(workspace.x) == length(rhs)
                function apply_diagonal!(y, x, alpha, beta)
                    if iszero(beta)
                        y .= alpha .* diagonal .* x
                    else
                        y .= alpha .* diagonal .* x .+ beta .* y
                    end
                    return y
                end
                op = Jutul.LinearOperator(Float64, 8, 8, false, false,
                    apply_diagonal!)
                Jutul.Krylov.krylov_solve!(workspace, op, rhs; history = true)
                @test workspace.stats.solved
                @test norm(diagonal .* workspace.x - rhs) < 1e-7
            end
        end
    end
end
