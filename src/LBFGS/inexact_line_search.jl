"""
    inexact_line_search(u0, v0, g0, d, f; kwargs...) -> (found_improvement, u, v, g, info)

Perform an inexact line search using strong Wolfe conditions with cubic and/or quadratic 
interpolation/extrapolation accounting for occasionally failed function evaluations and 
moderately noisy/inaccurate function values/gradients.

# Arguments
- `u0`: Initial point
- `v0`: Function value at initial point
- `g0`: Gradient at initial point
- `d`: Search direction (must be a descent direction, i.e., d'*g0 < 0)
- `f`: Function that evaluates objective value and gradient at a point u, returns (v, g)

# Keyword Arguments
- `max_it=5`: Maximum number of line search iterations
- `wolfe1=1e-4`: Armijo/sufficient decrease parameter (first Wolfe condition)
- `wolfe2=0.9`: Curvature condition parameter (strong second Wolfe condition)
- `max_step_increase=10.0`: Maximum factor by which step size can increase per iteration
- `max_step=1.0`: Maximum allowed step size (initial step size will be min(1.0, max_step))
- `step_diff_tol=1e-3`: Tolerance for considering two step sizes equal. Accordingly, any new
                        trial step is forced to be step_diff_tol away from existing evaluated points.
                        Function will return if bracketing interval becomes smaller 2x this tolerance.
- `value_diff_tol=sqrt(eps())`: Relative tolerance for considering function values equal
- `reduction_factor_failure=0.25`: Factor by which to reduce step size on failed evaluation
                                   NOTE: if failure occurs after successful improvement, we return 
                                   the best found evaluation instead of reducing step size
- `verbosity=1`: Verbosity level : 0=silent, only prints warnings
                                   1=minimal : only prints warnings and output from iteration 2 and onwards
                                   2=detailed : prints all iterations and line-search messages

# Returns
- `found_improvement`: Boolean indicating whether any improvement over initial point was found
- `u`: Best point found during line search
- `v`: Function value at best point
- `g`: Gradient at best point
- `info`: NamedTuple with fields:
  - `flag`: Boolean indicating if strong Wolfe conditions were satisfied
  - `step`: Final step size used
  - `nits`: Number of iterations performed
  - `values`: Vector of all objective values evaluated

# Description
The algorithm attempts to find a step size a satisfying the strong Wolfe conditions:
1. Sufficient decrease: f(u0 + a*d) ≤ f(u0) + wolfe1*a*d'*g0
2. Curvature: |d'*g(u0 + a*d)| ≤ wolfe2*|d'*g0|

The search uses bracketing with cubic and/or quadratic polynomial interpolation/extrapolation
roughly following (More & Thuente, 94) to efficiently locate acceptable step sizes. 
If function evaluations fail or Wolfe conditions cannot be satisfied within [0, max_step] or
within max_it iterations, the best evaluation found so far is returned.

# Notes
- The search direction d must satisfy d'*g0 < 0 (descent direction)
- If no improvement is found, initial point is returned with found_improvement=false
- Handles non-finite function values by reducing step size or returning early if
  improvement has already been found for another step size.
- The algorithm is hopefully quite robust wrt slightly noisy/inaccurate function evaluations
  and gradients, but this can be investigated further
"""

function inexact_line_search(u0, v0, g0, d, f;
                max_it = 5,
                wolfe1 = 1e-4, 
                wolfe2 = 0.9,
                max_step_increase = 10.0,
                max_step = 1.0,
                step_diff_tol = 1e-3,
                value_diff_tol = sqrt(eps()),
                reduction_factor_failure = 0.25,
                verbosity = 1)
    @assert d'*g0 < 0 "Line-search: Search direction is not a descent direction"
    # assign points
    p0 = (a=0.0, v=v0, g=d'*g0)
    print_iteration(verbosity, 0, p0)
    p_max = (a=max_step, v=NaN, g=NaN)
    p1, p2 = p0, p_max
    # Wolfe conditions
    w1 = (p) -> p.v <= p0.v + wolfe1 * p.a * p0.g
    w2 = (p) -> abs(p.g) <= wolfe2 * abs(p0.g)
    # initialize
    a = min(1.0, max_step)
    p = (a = a, v = NaN, g = NaN)
    # keep track of best point
    best_eval = (u = u0, v = v0, g = g0, a = 0.0)
    ls_done, wolfe_ok, at_max_step = false, false, false
    # tolerance for considering function values equal
    equal_value_tol = v0 * value_diff_tol
    it, msg, values = 0, "", [] 

    while !ls_done && it < max_it
        it += 1
        u = u0 + a * d
        v, g = f(u)
        if !isfinite(v)
            # function evaluation failed, we return if improvement already has been obtained,
            # otherwise reduce step size by reduction_factor_failure
            a, p2, p_max, ls_done = handle_failed_evaluation!(a, p0, p1, p2, p_max, best_eval, step_diff_tol, reduction_factor_failure, equal_value_tol)
            msg = print_warning_failed(a, ls_done) 
            continue
        end
        # update values/best point
        push!(values, v)
        if v < best_eval.v
            best_eval = (u = u, v = v, g = g, a = a)
        end
        # new current point
        p = (a=a, v=v, g=d'*g)
        at_max_step = abs(a - p_max.a) < step_diff_tol
        if at_max_step
            # update p_max with v, g
            p_max = p
        end
        is_wolfe1, is_wolfe2 = w1(p), w2(p)
        print_iteration(verbosity, it, p; wolfe = [is_wolfe1, is_wolfe2])
        # check Wolfe conditions
        if is_wolfe1 && is_wolfe2
            ls_done, wolfe_ok = true, true
            continue
        end
        # check if we are at max step (without Wolfe conditions), but current best and negative gradient
        if at_max_step && abs(p.v - best_eval.v) < equal_value_tol && p.g < 0
            # best is at max step and gradient is still negative
            ls_done, wolfe_ok = true, false
            continue
        end
        # update bracketing points
        if p.a > p2.a + step_diff_tol
            # extrapolation (p2 has been evaluated and p2.v < p1.v)
            p1, p2 = p2, p
        elseif p.v > p1.v || !isfinite(p2.v)
            # if p2 not evaluated yet, always set p2 = p so we can interpolate/extrapolate
            p2 = p
        else
            if p.g < 0.0
                p1 = p
            else
                p2 = p
            end
        end
        # compute next step - limit to max step increase
        a_max = min(max_step, max_step_increase * p2.a)
        if a_max < p_max.a - step_diff_tol
            p_max_cur = (a = a_max, v = NaN, g = NaN)
        else
            p_max_cur = p_max
        end
        a, msg = next_step(p1, p2, p_max_cur, step_diff_tol)

        if verbosity > 1 && msg != ""
            @printf("---  Line-search info: %s\n", msg)
        end
        
        if !isfinite(a)
            @warn("Line-search: returning, step selection failed with message: $msg.")
            ls_done = true
            continue
        end
    end
    found_improvement = abs(p0.v - best_eval.v) > equal_value_tol
    if !found_improvement
        best_eval = (u = u0, v = v0, g = g0, a = 0.0)
        @warn("Line-search: No improvement found, returning initial point.")
    end
    print_end_message(verbosity, it, best_eval, wolfe_ok, at_max_step, max_it, msg)
    info = (flag = wolfe_ok, step = best_eval.a, nits = it, objVals = values)
    return (found_improvement, best_eval.u, best_eval.v, best_eval.g, info)
end


function next_step(p1, p2, p_max, step_diff_tol)
    # assumes both p1 and p2 have been evaluated successfully, not neccarily p_max
    @assert isfinite(p1.v) && isfinite(p2.v) "Line search: next_step called with unevaluated points."
    # we scale to O(1):
    # g(s) = (f(a1 + s*(a2-a1)) - f(a1)) / ((a2-a1)|g1|), s in [0,1]
    # ps1 = (a = 0.0, v = 0.0, g = -1.0)
    ps2 = (a = 1.0, 
           v = ((p2.v - p1.v) / (p2.a - p1.a)) / abs(p1.g), 
           g = p2.g / abs(p1.g))
    unscale_arg = (a_scaled) -> p1.a + a_scaled * (p2.a - p1.a)
    a, msg = NaN, ""
    if ps2.v >= 0 || ps2.g >= 0
        # interpolation within p1 and p2
        if ps2.v >= 0 
            a = unscale_arg( max(cubicmin(ps2), quadmin1(ps2)) )
        else #ps2.g >= 0 && ps2.v < 0
            a = unscale_arg( max(cubicmin(ps2), quadmin2(ps2)) )
        end
        # check that we're not too close to p1 or p2 
        if a < p1.a + step_diff_tol ||  a > p2.a - step_diff_tol || !isfinite(a)
            a, msg = ad_hoc_step(p1, p2, step_diff_tol)
            if !isfinite(a)
                return NaN, msg
            end
        end
    else
        # extrapolation above p2, be more careful (p2 has been evaluated, but probably not p_max)
        if ps2.g > -1 + sqrt(eps())
            # flattening out, might have cubic minimum, but choose to trust quadratic more
            a = unscale_arg(quadmin2(ps2))
        else 
            # gradient is getting steeper, try maximal allowed step
            a = p_max.a
        end
        if a > p_max.a - step_diff_tol || !isfinite(a)
            # check if we have been at a_max before
            if isfinite(p_max.v)
                # do ad-hoc step-selection
                a, msg = ad_hoc_step(p2, p_max, step_diff_tol)
                if !isfinite(a)
                    return NaN, msg
                end
            else
                a = p_max.a
            end
        else
            # check that we're not too close to p2 
            if a < p2.a + step_diff_tol
                a, msg = ad_hoc_step(p2, p_max, step_diff_tol)
                if !isfinite(a)
                    return NaN, msg
                end
            end
        end
    end
    return a, msg
end

function cubicmin(p2)
    # step corresponding to minimum of scaled/shifted cubic polynomial 
    # using values and derivatives at 0 and 1
    # c0 = 0, c1 = -1
    c2 = 3*p2.v - p2.g + 2
    c3 = -2*p2.v + p2.g - 1
    if abs(c3) < sqrt(eps())
        return quadmin1(p2)
    end
    r = c2^2 + 3*c3
    if r < sqrt(eps())
        return NaN
    end
    return (-c2 + sqrt(r)) / (3*c3)
end

function quadmin1(p2)
    # step corresponding to minimum of scaled/shifted quadratic polynomial
    # using values at 0, 1 and derivative at 0
    # c0 = 0, c1 = -1
    c2 = p2.v + 1
    if c2 <= sqrt(eps())
        return NaN
    end
    return 1 / (2 * c2)
end

function quadmin2(p2)
    # step corresponding to minimum of scaled/shifted quadratic polynomial
    # using values at 0, 1 and derivative at 1
    # c0 = 0
    c1 = 2*p2.v - p2.g
    c2 = -p2.v + p2.g
    if c2 <= sqrt(eps())
        return NaN
    end
    return -c1 / (2 * c2)
end

function ad_hoc_step(p1, p2, step_diff_tol)
    # take weighted mean with emphasis on smaller function value
    low, high = p1.a + step_diff_tol, p2.a - step_diff_tol
    if high <= low
        # interval too narrow
        msg = @sprintf("Interval boundary [%4.4f, %4.4f] too narrow.", p1.a, p2.a)
        return NaN, msg
    end
    if p1.v < p2.v
        a = 2*low/3 + high/3
    else
        a = low/3 + 2*high/3
    end
    msg = @sprintf("Step too close to interval boundary [%4.4f, %4.4f], re-setting to %4.4f.", p1.a, p2.a, a)
    return a, msg
end


function handle_failed_evaluation!(a, p0, p1, p2, p_max, best, step_diff_tol, reduction_factor_failure, equal_value_tol)
    # if we have successfully evaluated any improvement, we return that and end the line-search
    if abs(p0.v - best.v) > equal_value_tol
        a = best.a
        ls_done = true
        return a, p2, p_max, ls_done # points not relevant anymore
    end
    # we may have two situations:
    # 1) p2 has successfully been evaluated and a < p2.a with p2.v > p1.v = p0.v
    # 2) we have no successful evaluations (except p0)
    # in either case we cap maximial a-value according to reduction factor
    a = a * reduction_factor_failure
    if a <= step_diff_tol
        # can't reduce further, give up
        a = NaN
        ls_done = true
    else
        p_max = (a = a, v = NaN, g = NaN)
        p2 = p_max
        ls_done = false
    end
    return a, p2, p_max, ls_done
end


function print_iteration(verbosity, it, p; wolfe = [nothing, nothing])
    if verbosity == 0 || (it < 2 && verbosity == 1)
        return
    end
    if isnothing(wolfe[1]) || isnothing(wolfe[2])
        @printf("  Line-search - %2d | step = %6.3e | v = %11.3e | dvdd = %11.3e \n", it, p.a, p.v, p.g)
    else
        @printf("  Line-search - %2d | step = %6.3e | v = %11.3e | dvdd = %11.3e | wolfe ( %1d, %1d)\n", it, p.a, p.v, p.g, wolfe[1], wolfe[2])
    end
end


function print_end_message(verbosity, it, best, wolfe_ok, at_max, ls_max_iter, msg)
    if verbosity <= 1 
        return
    end
    if wolfe_ok
        @printf("Line-search succeeded in %d iterations, step = %6.3f, v = %11.3e\n", it, best.a, best.v)
    elseif it >= ls_max_iter
        @printf("Line-search reached maximum iterations (%d), step = %6.3f, v = %11.3e\n", it, best.a, best.v)
    elseif at_max
        @printf("Line-search stopped at maximum step size (%6.3f), step = %6.3f, v = %11.3e\n", best.a, best.a, best.v)
    else
        @printf("Line-search stopped after %d iterations with message: %s, step = %6.3f, v = %11.3e\n", it, msg, best.a, best.v)
    end
end

function print_warning_failed(a, ls_done)
    msg = ""
    if ls_done
        if !isfinite(a)
            @warn("Line-search: Function evaluation failed and step size cannot be reduced further.")
            msg = "function evaluation failed at minimal step size"
        else
            @warn(@sprintf("Line-search: Function evaluation failed, returning best evaluated value at step size a = %4.4f.", a))
            msg = "function evaluation failed, returning best evaluated value"
        end
    else
        @warn(@sprintf("Line-search: Function evaluation failed, reducing maximal step size to a = %4.4f.", a))
    end
    return msg
end
