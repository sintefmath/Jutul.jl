using Jutul, Test, ForwardDiff
import Jutul: unit_box_bfgs
import Jutul.LBFGS: box_bfgs

function Rosenbrock(u0, lb, ub, n; scale = true)
    @assert length(u0) == n
    if scale
        u0 = u0 .* (ub - lb) + lb
    end
    f = 0
    g = zeros(n)
    rosenbrock_i(u) = (1.0 - u[1])^2 + 100.0 * (u[2] - u[1]^2)^2
    for i in 1:2:n
        u_i = u0[i:(i+1)]
        fval = rosenbrock_i(u_i)
        g1, g2 = ForwardDiff.gradient(rosenbrock_i, u_i)

        g1_a = -2.0 * (1.0 - u0[i]) - 400.0 * (u0[i + 1] - u0[i]^2) * u0[i]
        g2_a = 200.0 * (u0[i + 1] - u0[i]^2)

        @test g1 ≈ g1_a
        @test g2 ≈ g2_a
        f += fval
        g[i] = g1
        g[i + 1] = g2
    end
    if scale
        for i in 1:n
            g[i] = g[i] * (ub[i] - lb[i])
        end
    end
    return (f, g)
end

@testset "LBFGS Rosenbrock" begin
    # Test minimizer
    n = 10
    lb = repeat([-100], n)
    ub = repeat([100], n)
    x0 = collect(range(-100, stop = 100, length = n))
    u0 = (x0 - lb) ./ (ub - lb)
    f! = (u) -> Rosenbrock(u, lb, ub, n)
    v, u, history = unit_box_bfgs(u0, f!; maximize = false, print = 0)
    u = u .* (ub - lb) + lb
    @test history.val[end] < 160

    # Test negative maximize
    f! = (u) -> (-1, -1) .* Rosenbrock(u, lb, ub, n)
    v, u, history = unit_box_bfgs(u0, f!; maximize = true, print = 0)
    u = u .* (ub - lb) + lb
    @test history.val[end] > -151

    f! = (u) -> Rosenbrock(u, lb, ub, n, scale = false)
    v, x, history = box_bfgs(x0, f!, lb, ub; maximize = false, print = 0)
    @test history.val[end] < 160

    f! = (u) -> (-1, -1) .* Rosenbrock(u, lb, ub, n, scale = false)
    v, x, history = box_bfgs(x0, f!, lb, ub; maximize = true, print = 0)
    @test history.val[end] < 160
end

@testset "LBFGS linear constraints" begin
    # Equality constraint
    n = 4
    lb = repeat([-100.0], n)
    ub = repeat([100.0], n)
    x0 = [3.0, 1.0, 1.0, 1.0]
    A = [1.0, -1.0, -1.0, -1.0]'
    lin_eq = (A = A, b = [0.0])
    f! = (u) -> Rosenbrock(u, lb, ub, n, scale = false)
    v, x, history = box_bfgs(x0, f!, lb, ub; lin_eq = lin_eq, maximize = false, print = 0)
    @test x[1] ≈ sum(x[2:end])

    x0 = [3.0, 3.0, 2.0, 1.0]
    A = [1.0 -1.0 0.0 0.0;
         0.0 0.0 1.0 -1.0]
    lin_eq = (A = A, b = [0.0, 1.0])
    v, x, history = box_bfgs(x0, f!, lb, ub; lin_eq = lin_eq, maximize = false, print = 0)
    @test x[1] ≈ x[2]
    @test x[3] ≈ x[4] + lin_eq.b[2]

    # Inequality constraint
    n = 2
    lb = repeat([-100.0], n)
    ub = repeat([100.0], n)
    x0 = [1.0, 5.0]
    A = [0.8, -1.0]'
    lin_ineq = (A = A, b = [0.0])
    f! = (u) -> Rosenbrock(u, lb, ub, n, scale = false)
    v, x, history = box_bfgs(x0, f!, lb, ub; lin_ineq = lin_ineq, maximize = false, print = 0)
    @test 0.8 * x[1] - x[2] <= 1e-5

    x0 = [10.0, 2.0]
    A = [-1.0, 2.0]'
    lin_ineq = (A = A, b = [-5.0])
    v, x, history = box_bfgs(x0, f!, lb, ub; lin_ineq = lin_ineq, maximize = false, print = 0)
    @test -1.0 * x[1] + 2.0 * x[2] <= lin_ineq.b[1] + 1e-5
end
