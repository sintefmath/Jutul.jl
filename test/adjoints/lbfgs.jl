using Jutul, Test
import Jutul: unit_box_bfgs

function Rosenbrock(u0, lb, ub, n)
    @assert length(u0) == n
    u0 = u0 .* (ub - lb) + lb
    f = 0
    g = zeros(n)
    for i in 1:2:n
        f += (1.0 - u0[i])^2 + 100.0 * (u0[i + 1] - u0[i]^2)^2
        g[i] = -2.0 * (1.0 - u0[i]) - 400.0 * (u0[i + 1] - u0[i]^2) * u0[i]
        g[i + 1] = 200.0 * (u0[i + 1] - u0[i]^2)
    end
    for i in 1:n
        g[i] = g[i] * (ub[i] - lb[i])
    end
    return (f, g)
end

@testset "LBFGS Rosenbrock" begin
    # Test minimizer
    n = 10
    lb = repeat([-100], n)
    ub = repeat([100], n)
    u0 = collect(range(-100, stop = 100, length = n))
    u0 = (u0 - lb) ./ (ub - lb)
    f! = (u) -> Rosenbrock(u, lb, ub, n)
    v, u, history = unit_box_bfgs(u0, f!; maximize = false, print = 0)
    u = u .* (ub - lb) + lb
    @test history.val[end] < 160

    # Test negative maximize
    f! = (u) -> (-1, -1) .* Rosenbrock(u, lb, ub, n)
    v, u, history = unit_box_bfgs(u0, f!; maximize = true, print = 0)
    u = u .* (ub - lb) + lb
    @test history.val[end] > -151
end
