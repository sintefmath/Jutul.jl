export newton_step
using Printf

function newton_step(model, storage; dt = nothing, linsolve = nothing, sources = nothing, iteration = nan)
    update_equations!(model, storage, dt = dt, sources = sources)
    update_linearized_system!(model, storage)
    lsys = storage["LinearizedSystem"]

    e = norm(lsys.r, Inf)
    @printf("It %d: |R| = %e\n", iteration, e)

    solve!(lsys)

    storage["state"]["Pressure"] += lsys.dx
    tol = 1e-6
    return (e, tol)
end
