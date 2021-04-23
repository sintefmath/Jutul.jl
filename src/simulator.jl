export newton_step
using Printf

function newton_step(model, storage; dt = nothing, linsolve = nothing, sources = nothing, iteration = nan)
    t_asm = @elapsed update_equations!(model, storage, dt = dt, sources = sources)
    @debug "Assembled equations in $t_asm seconds."
    t_lsys = @elapsed update_linearized_system!(model, storage)
    @debug "Updated linear system in $t_lsys seconds."

    lsys = storage["LinearizedSystem"]

    e = norm(lsys.r, Inf)
    @printf("It %d: |R| = %e\n", iteration, e)

    solve!(lsys, linsolve)

    storage["state"]["Pressure"] += lsys.dx
    tol = 1e-6
    return (e, tol)
end

# function simulate(model, state0, timesteps; dt = nothing, linsolve = nothing, sources = nothing, iteration = nan)
#
# end