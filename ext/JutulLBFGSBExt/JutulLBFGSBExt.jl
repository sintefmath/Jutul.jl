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
            maxfun = 4*max_it,
            maximize = false,
            scale = true,
            kwarg...
        )
        F = Jutul.DictOptimization.setup_optimization_functions(problem, maximize = maximize, scale = scale)
        _, x = LBFGSB.lbfgsb(F.f, F.g, F.x0;
            lb = F.min,
            ub = F.max,
            iprint = 1,
            factr = 1.0/obj_change_tol,
            pgtol = grad_tol,
            maxiter = max_it,
            maxfun = maxfun,
            kwarg...
        )
        return (F.descale(x), F.history)
    end
end
