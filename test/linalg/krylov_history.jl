using Jutul, LinearAlgebra, Test

mutable struct KrylovHistoryCountMatrix <: AbstractMatrix{Float64}
    values::Matrix{Float64}
    products::Int
end
Base.size(A::KrylovHistoryCountMatrix) = size(A.values)
Base.getindex(A::KrylovHistoryCountMatrix, i::Int, j::Int) = A.values[i, j]
function LinearAlgebra.mul!(y::AbstractVector, A::KrylovHistoryCountMatrix,
        x::AbstractVector, alpha::Number, beta::Number)
    A.products += 1
    return mul!(y, A.values, x, alpha, beta)
end
LinearAlgebra.mul!(y::AbstractVector, A::KrylovHistoryCountMatrix, x::AbstractVector) =
    mul!(y, A, x, 1.0, 0.0)

function solve_history_case(values, rhs; max_iterations = 10, min_iterations = 1)
    A = KrylovHistoryCountMatrix(copy(values), 0)
    r, dx = copy(rhs), zeros(length(rhs))
    sys = Jutul.LinearizedSystem(A, r, dx, Float64[], r, dx, Jutul.EquationMajorLayout())
    model = SimulationModel(ScalarTestDomain(), ScalarTestSystem())
    solver = GenericKrylov(:gmres; relative_tolerance = 1e-12,
        max_iterations, min_iterations)
    result = Jutul.linear_solve!(sys, solver, model.context, model)
    return (; result.ok, result.iterations, A.products, x = copy(solver.storage.x), dx)
end

@testset "Krylov residual history avoids redundant products" begin
    identity = Matrix{Float64}(I, 3, 3)
    rhs = [1.0, 0.0, 0.0]
    single = solve_history_case(identity, rhs)
    @test single.ok
    @test single.iterations == 1
    @test single.products == 1
    @test single.x == rhs
    @test single.dx == -rhs

    # These paths retain the explicit true-residual product.
    initial = solve_history_case(identity, zeros(3))
    @test initial.ok && initial.iterations == 0
    @test initial.products == 1
    @test iszero(initial.x) && iszero(initial.dx)

    diagonal = Matrix(Diagonal([1.0, 2.0, 3.0]))
    failed = solve_history_case(diagonal, ones(3); max_iterations = 1)
    @test !failed.ok && failed.iterations == 1
    @test failed.products == 2
    @test all(isfinite, failed.x) && failed.dx == -failed.x

    manual = solve_history_case(identity, rhs; min_iterations = 2, max_iterations = 1)
    @test manual.iterations == 1
    @test manual.products == 2
    @test manual.x == single.x && manual.dx == single.dx
end
