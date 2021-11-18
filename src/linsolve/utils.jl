export IterativeSolverConfig, reservoir_linsolve

mutable struct IterativeSolverConfig
    relative_tolerance
    absolute_tolerance
    max_iterations
    nonlinear_relative_tolerance
    relaxed_relative_tolerance
    verbose
    arguments
    function IterativeSolverConfig(;relative_tolerance = 1e-3,
                                    absolute_tolerance = nothing, 
                                    max_iterations = 100,
                                    verbose = false,
                                    nonlinear_relative_tolerance = nothing,
                                    relaxed_relative_tolerance = 0.1,
                                    kwarg...)
        new(relative_tolerance, absolute_tolerance, max_iterations, nonlinear_relative_tolerance, relaxed_relative_tolerance, verbose, kwarg)
    end
end

function reservoir_linsolve(model, method = :cpr; rtol = nothing, v = 0, provider = Krylov, solver = Krylov.bicgstab)
    if model.context == DefaultContext()
        return nothing
    end
    if method == :cpr
        gs_its = 1
        cyc = AlgebraicMultigrid.V()
        p_solve = AMGPreconditioner(ruge_stuben, cycle = cyc, presmoother = GaussSeidel(iter = gs_its), postsmoother = GaussSeidel(iter = gs_its))
        # p_solve = AMGPreconditioner()
        #p_solve = AMGPreconditioner(smoothed_aggregation, cycle = cyc, 
        #                        smooth = JacobiProlongation(2.0/3.0), 
        #                        presmoother = GaussSeidel(iter = gs_its), postsmoother = GaussSeidel(iter = gs_its))
        # p_solve = AMGPreconditioner(ruge_stuben)
        # p_solve = AMGPreconditioner(smoothed_aggregation)

        cpr_type = :true_impes
        cpr_type = :analytical
        update_interval = :iteration
        update_interval = :ministep
        update_interval = :once

        prec = CPRPreconditioner(p_solve, strategy = cpr_type, 
        update_interval = update_interval, partial_update = update_interval == :once)
        rtol = isnothing(rtol) ? 0.001 : rtol
        max_it = 2
        max_it = 200
        # max_it = 10
    elseif method == :ilu0
        prec = ILUZeroPreconditioner(right = false)
        rtol = isnothing(rtol) ? 0.001 : rtol
        max_it = 200
    else
        return nothing
    end
    atol = 1e-12
    # v = -1
    # v = 0
    # v = 1
    # if true
    #     krylov = Krylov.bicgstab
    #     # krylov = Krylov.gmres
    #     pv = Krylov
    # else
    #     krylov = IterativeSolvers.gmres!
    #     krylov = IterativeSolvers.bicgstabl!
    #     pv = IterativeSolvers
    # end
    nl_reltol = 1e-3
    relaxed_reltol = 0.25

    lsolve = GenericKrylov(solver, provider = provider, verbose = v, preconditioner = prec, 
            nonlinear_relative_tolerance = nl_reltol,
            relaxed_relative_tolerance = relaxed_reltol,
            relative_tolerance = rtol, absolute_tolerance = atol,
            max_iterations = max_it)
    return lsolve
end
