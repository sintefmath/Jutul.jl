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
        ub = problem.limits.max
        lb = problem.limits.min
        x0 = problem.x0
        # Use local variables to handle caching
        prev_hash = NaN
        prev_val = NaN
        prev_grad = similar(ub)
        history = Float64[]
        function feval(x, dFdx = missing)
            hash_x = hash(x)
            if prev_hash == hash_x
                obj = prev_val
            else
                f, g = problem(x; gradient = true)
                if maximize
                    f = -f
                    @. g = -g
                end
                prev_val = obj = f
                prev_grad .= g
                prev_hash = hash_x
                push!(history, obj)
            end
            if !ismissing(dFdx)
                dFdx .= prev_grad
            end
            return obj
        end
        function f!(x)
            return feval(x)
        end
        function g!(dFdx, x)
            feval(x, dFdx)
            return dFdx
        end

        _, x = LBFGSB.lbfgsb(f!, g!, x0; lb=lb, ub=ub,
            iprint = 0,
            factr = 1.0/obj_change_tol,
            pgtol = grad_tol,
            maxfun = 100,
            maxiter = max_it,
            kwarg...
        )
        return (x, history)
    end
end
