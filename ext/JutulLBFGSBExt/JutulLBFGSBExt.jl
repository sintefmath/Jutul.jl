module JutulLBFGSBExt
    using Jutul
    import LBFGSB

    function Jutul.check_lbfgsb_availability_impl()
        return true
    end

    function Jutul.DictOptimization.optimize_implementation(problem, ::Val{:lbfgsb};
            grad_tol = 1e-6,
            obj_change_tol = 1e-6,
            max_it = 25,
            maximize = false,
            kwarg...
        )
        F = Jutul.DictOptimization.setup_optimzation_functions(problem, maximize = maximize)

        _, x = LBFGSB.lbfgsb(F.f, F.g, F.x0; lb=F.min, ub=F.max,
            iprint = 0,
            factr = 1.0/obj_change_tol,
            pgtol = grad_tol,
            maxfun = 100,
            maxiter = max_it,
            kwarg...
        )
        return (F.descale(x), F.history)
    end
end
