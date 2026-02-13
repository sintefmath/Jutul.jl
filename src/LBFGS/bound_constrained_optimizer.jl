"""
    optimize_bound_constrained(u0, f, lb, ub; kwargs...)

Iterative line search optimization using L-BFGS for bound constrained problems using 
exact qp-solves for the search direction.

Partial translation/improvement of the MRST function `optimizeBoundConstrained`.

# Arguments
- `u0`: Initial guess vector of length n with 0 ≤ u0 ≤ 1, must be feasible
- `f`: Function handle `f(x)` that returns tuple `(v, g)` where:
    * `v`: objective value
    * `g`: objective gradient vector of length n
- `lb`: Lower bound(s) on u
- `ub`: Upper bound(s) on u 

# Keywords
- `maximize::Bool=false`: Set to true to maximize objective, false to minimize (default)
- `step_init::Float64=NaN`: Initial step (gradient scaling). If NaN, uses `max_initial_update/max(|initial gradient .* (ub-lb)|)`
- `max_initial_update::Float64=0.1`: Relative maximal initial update step
- `history::Any=nothing`: For warm starting based on previous optimization (requires `output_hessian=true`)

## Stopping Criteria
- `grad_tol::Float64=-Inf`: Absolute tolerance of inf-norm of projected gradient
- `grad_rel_tol::Float64=1.0e-4`: Relative gradient tolerance (sets `grad_tol` if finite)
- `obj_tol::Float64=-Inf`: Absolute tolerance on objective value (returns if abs(objective) < tol)
- `obj_rel_tol::Float64=1e-4`: Relative tolerance on objective value (sets `obj_tol` if finite)
- `obj_change_tol::Float64=-Inf`: Absolute tolerance on objective change between iterations
- `obj_change_tol_rel::Float64=1.0e-7`: Relative tolerance on objective change (sets `obj_change_tol` if finite)
- `max_it::Int=25`: Maximum number of iterations
- Line search failure also triggers termination (after Hessian reset attempts)

## Line Search Options)
- `use_new_line_search::Bool=false`: Whether to use new line search implementation (from inexact_line_search.jl) 
   or old one (from constrained_optimizer.jl). Some options are only relevant for the new line search.
- `ls_max_it::Int=5`: Maximum number of line-search iterations
- `ls_wolfe1::Float64=1e-4`: Objective improvement condition (Wolfe condition 1)
- `ls_wolfe2::Float64=0.9`: Gradient reduction condition (Wolfe condition 2)
- `ls_max_step_increase::Float64=10.0`: Maximum step increase factor between line search iterations
- `ls_step_diff_tol::Float64=1e-3`: Step difference tolerance for line search
- `ls_reduction_factor_failure::Float64=0.3`: Reduction factor applied on line search failure
- `ls_verbosity::Int=1`: Verbosity level for line search output
- `ls_safeguard_fac::Float64=1e-5`: Safeguard factor for line search

## Hessian Approximation (L-BFGS)
- `lbfgs_num::Int=5`: Number of vector-pairs stored for L-BFGS
- `lbfgs_strategy::Symbol=:dynamic`: Strategy for L-BFGS (:static or :dynamic)
- `lbfgs_require_wolfe::Bool=false`: Require Wolfe conditions for BFGS update

## QP Solve Options
- `max_it_qp::Int=50`: Maximum iterations for solving QP problem
- `active_chunk_tol::Float64=sqrt(eps())`: Tolerance for chunking active constraints

## Trust Region Options
- `use_trust_region::Bool=false`: Use infinity-norm trust region
- `radius_increase::Float64=2.0`: Update factor for 'good' approximations
- `radius_decrease::Float64=0.25`: Update factor for 'bad' approximations
- `ratio_thresholds::Vector{Float64}=[0.25, 0.75]`: Thresholds for trust region updates
- `trust_region_init::Float64=NaN`: Initial trust region radius (defaults to `max_initial_update`)

## Other Options
- `scale_to_unit_box::Bool=false`: Scale problem to unit box [0, 1]
- `output_hessian::Bool=false`: Include Hessian approximation in history output

# Returns
- `v`: Optimal or best objective value
- `u`: Control/parameter vector corresponding to v
- `history`: Named tuple containing iteration history
"""

function optimize_bound_constrained(
        u0, f, lb, ub;
        maximize = false,
        step_init = NaN,
        max_initial_update = 0.1,
        obj_tol = -Inf,
        obj_rel_tol = 1e-4,
        grad_tol = -Inf,
        grad_rel_tol = 1.0e-4,
        obj_change_tol = -Inf,
        obj_change_tol_rel = 1.0e-7,
        max_it = 25,
        use_new_line_search = true,
        ls_max_it = 5,
        ls_wolfe1 = 1.0e-4,
        ls_wolfe2 = 0.9,
        ls_max_step_increase = 10.0,
        ls_step_diff_tol = 1.0e-4,
        ls_reduction_factor_failure = 0.3,
        ls_verbosity = 1,
        ls_safeguard_fac = 1.0e-5,
        max_it_qp = 250,
        active_chunk_tol = sqrt(eps()),
        lbfgs_num = 5,
        lbfgs_strategy = :dynamic,
        lbfgs_require_wolfe = false,
        use_trust_region = false,
        trust_region_init = NaN,
        radius_increase = 2.0,
        radius_decrease = 0.25,
        ratio_thresholds = [0.25, 0.75],
        scale = false,
        output_hessian = false,
        history = nothing
    )
    
    # Negate f if we are maximizing
    obj_sign = 1
    if maximize
        f! = (u) -> f_negative(u, f)
        obj_sign = -1
    else
        f! = (u) -> f(u)
    end
    # handle bounds
    lb, ub = handle_bounds!(lb, ub, length(u0))
    # Potential scaling
    if scale
        f_old! = f!
        f! = (u) -> f_scale(u, f_old!, lb, ub)
        u0 = (u0 .- lb) ./ (ub .- lb)
        lb, ub = zeros(length(u0)), ones(length(u0))
    end
    # Initialize variables
    if isnothing(history) # starting from scratch
        if any(lb .> u0) || any(u0 .> ub)
            @warn "Initial guess was not within bounds, projecting to feasible domain."
            u0 = max.(lb, min.(ub, u0))
        end
        # Perform initial evaluation of objective and gradient
        v0, g0 = f!(u0)
        if !isfinite(v0)
            error("Initial objective is non-finite.")
        end
        # If not provided, set initial step
        step = step_init
        if isnan(step) || step <= 0
            step = max_initial_update / maximum(abs.(g0 .* (ub .- lb)))
        end
        # Update absolute stopping tolerances based on relative
        if isfinite(obj_change_tol_rel) && obj_change_tol_rel > 0
            obj_change_tol = max(obj_change_tol_rel * abs(v0), obj_change_tol)
        end
        if isfinite(grad_rel_tol) && grad_rel_tol > 0
            grad_tol = max(grad_rel_tol * norm(g0, Inf), grad_tol)
        end
        if isfinite(obj_rel_tol) && obj_rel_tol > 0
            obj_tol = max(obj_rel_tol * abs(v0), obj_tol)
        end
        # Initialize trust region radius
        r_trust = trust_region_init
        if use_trust_region && isnan(r_trust)
            r_trust = max_initial_update
        end
        # Initialize Hessian approximation with scaling based on initial step
        H = LimitedMemoryHessian(init_scale = 1/step, m = lbfgs_num, init_strategy = lbfgs_strategy)
        H_prev = deepcopy(H)
        it = 0
    else # starting from previous optimization
        @assert false "Warm starting not yet fully implemented."
        it = length(history.val)
        H = history.hess[it]
        @assert !isnothing(H) && !isempty(H) "Warm start based on history requires Hessian approximations"
        u0 = history.u[it]
        v0, g0 = f!(u0)
        H_prev = history.hess[max(it - 1, 1)]
        max_it = max_it + it
        r_trust = history.r[it]
    end
    # Print info for iteration 0
    info = update_info!(nothing; obj_info = (v = obj_sign * v0, pg = norm(g0, Inf), n_active = 0))
    print_info_step(info)
    
    v, u, g = v0, copy(u0), copy(g0)
    n_active = 0
    success = false
    stop_flags = Dict(:grad => false, :obj_change => false, :obj => false,
                      :ls_fail => false, :maxit => false)
    stop_tols = (grad = grad_tol, obj_change = obj_change_tol, obj = obj_tol, 
                 ls_fail = true, maxit = max_it)
    while !any(values(stop_flags))
        it += 1
        # Determine current bounds based on trust region
        if !use_trust_region
            lb_cur, ub_cur = lb, ub
        else
            lb_cur, ub_cur = incorporate_trust_region(u0, r_trust, lb, ub)
        end
        # Get search direction by solving QP problem
        d, H, H_prev, pg, ls_max_step, qpinfo = get_search_direction_qp!(
            u0, g0, H, H_prev, lb_cur, ub_cur,
            grad_tol, max_it_qp, active_chunk_tol
        )

        if ls_max_step > 0.0
            if !use_new_line_search
                # Perform line-search (from constrained_optimizer.jl)
                u, v, g, lsinfo = line_search(
                    u0, v0, g0, d, f!;
                    wolfe1 = ls_wolfe1,
                    wolfe2 = ls_wolfe2,
                    safeguardFac = ls_safeguard_fac,
                    stepIncreaseTol = ls_max_step_increase,
                    line_searchmax_it  = ls_max_it,
                    maxStep = ls_max_step
                )
                ls_success = v < v0*(1 - 100*eps()) # somewhat ad-hoc
            else
                # Perform line-search (from inexact_line_search.jl)
                ls_success, u, v, g, lsinfo = inexact_line_search(
                        u0, v0, g0, d, f!;
                        max_it = ls_max_it,
                        wolfe1 = ls_wolfe1,
                        wolfe2 = ls_wolfe2,
                        max_step_increase = ls_max_step_increase,
                        max_step = ls_max_step,
                        step_diff_tol = ls_step_diff_tol,
                        verbosity = ls_verbosity,
                        reduction_factor_failure = ls_reduction_factor_failure
                )
            end
            if !ls_success
                # reset hessian approximation (if we haven't already tried this) and retry line search
                if !(H.it_count == 0)
                    H = reset!(H)
                    @warn("Line search failed, resetting Hessian approximation and restarting iteration.")
                else
                    stop_flags[:ls_fail] = true
                end
                continue
            end
            # predicted reduction in objective
            dobj_est = (u-u0)' * g0 + 0.5 * (u-u0)' * (H * (u-u0))
            dobj_true = v - v0
            # Compute trust region ratio (quadratic model fit)
            rho = dobj_true / dobj_est
            # Update trust region radius
            if use_trust_region
                r_trust = update_trust_region!(r_trust, rho, norm(u - u0, Inf), lsinfo.step, 
                                               radius_increase, radius_decrease, ratio_thresholds)
            else
                r_trust = NaN
            end
            # Check requirements for updating Hessian
            du, dg = u - u0, g - g0
            # do_update = du' * dg > sqrt(eps()) * norm(du) * norm(dg)
            do_update = du' * dg > sqrt(eps()) * norm(dg)^2
            if lbfgs_require_wolfe
                do_update = do_update && lsinfo.flag > 0
            end
            
            if do_update
                # If any of the gradient entries are not defined, set difference to zero
                dg[.!isfinite.(dg)] .= 0
                H_prev = deepcopy(H)
                H = update!(H, du, dg)
            else
                @printf("Hessian not updated during iteration %d.\n", it)
            end
            # update projected gradient
            active = get_active_bounds(u, -g, lb, ub)
            pg = proj_q(g, active)
            n_active = count(active)
            # Gather info
            tr_info = (r_trust = r_trust, rho = rho)
        else
            # No update -> no info
            lsinfo = qp_info = tr_info = nothing
        end
        obj_info = (v = obj_sign * v, pg = norm(pg, Inf), n_active = n_active)
        info = update_info!(info; obj_info = obj_info, qp_info = qpinfo, ls_info = lsinfo, tr_info = tr_info)
        
        # Check stopping criteria
        stop_flags[:grad] = norm(pg, Inf) < stop_tols.grad
        stop_flags[:obj] = abs(v) < stop_tols.obj
        stop_flags[:obj_change] = abs(v - v0) < stop_tols.obj_change
        stop_flags[:maxit] = it >= stop_tols.maxit
        stop_flags[:ls_fail] = !ls_success
        # Reset for next iteration
        v0, u0, g0 = v, copy(u), copy(g)
        
        print_info_step(info)
    end
    
    if scale
        u = u .* (ub .- lb) .+ lb
    end
    print_end_message(stop_flags, stop_tols, info)
    return (v, u, info)
end

function optimize_bound_constrained(problem; kwarg...)
    ub = problem.limits.max
    lb = problem.limits.min
    return optimize_bound_constrained(problem.x0, problem, lb, ub; kwarg...)
end

"""
Get search direction by solving QP problem iteratively
"""
function get_search_direction_qp!(u, g, H, H_prev, lb, ub, grad_tol, max_it_qp, active_chunk_tol)
    # Check whether projected gradient is below threshold
    active = get_active_bounds(u, -g, lb, ub)
    pg = g[.!active]
    if norm(pg, Inf) <= grad_tol
        d = zeros(size(u))
        max_step = 0.0
        qpinfo = nothing
        return (d, H, pg, max_step, qpinfo)
    end
    # In case of problematic Hessian approximation, we try up to 3 times
    rough_solve_info = nothing
    active_set_info = nothing
    success = false
    for trial_no in 1:3
        if trial_no == 2
            # try previous Hessian approximation
            H = deepcopy(H_prev)
        elseif trial_no == 3
            # reset to scaled identity (results in gradient direction)
            H = reset!(H)
            H_prev = reset!(H_prev)
        end
        # do the rough QP solve first
        d, g_rough, success, rough_solve_info = solve_rough_qp(u, g, H, lb, ub, 10)
        if !success
            # rough QP solve did not converge, repair last part using active-set method
            d_qp, g_qp, success, active_set_info = solve_active_set_qp(u .+ d, g_rough, H, lb, ub, max_it_qp, active_chunk_tol)
            d = d .+ d_qp
        else
            active_set_info  = (nits = 0, conv = true, nactive = rough_solve_info.nactive, nrelease = 0)
        end
        if !success
            @warn "Unable to solve local QP-problem in $max_it_qp iterations."
        end
        # Final projection just in case
        d = max.(lb, min.(ub, u .+ d)) .- u
        # Find max step size before hitting next bound
        _, max_step = find_next_bounds(u, d, falses(size(u)), lb, ub, 0.0)
        
        if max_step < 1 - sqrt(eps())
            @warn @sprintf("Problematic search direction, maximum step: %f < 1\n", max_step)
        end
        # Check if search direction is decreasing
        is_decreasing = d' * g <= 0
        if is_decreasing # decreasing search direction found, exit
            break
        else
            # Retry with other Hessian approx
            str = "Non-decreasing search direction"
            if trial_no == 1
                @printf("%s, trying previous Hessian approximation.\n", str)
            elseif trial_no == 2
                @printf("%s, trying to reset Hessian to identity.\n", str)
            elseif trial_no == 3
                @printf("Exiting: %s.\n", str)
                d = zeros(size(d))
                max_step = 0.0
            end
        end
    end
    qpinfo = (rough_solve = rough_solve_info, active_set = active_set_info, success = success)
    return (d, H, H_prev, pg, max_step, qpinfo)
end

function solve_active_set_qp(u0, g0, H, lb, ub, max_it, active_chunk_tol)
    # Solves corresponding QP-problem and returns d = u* - u0
    g = copy(g0)
    u = copy(u0)
    d = zeros(size(u))
    active = get_active_bounds(u, -g, lb, ub)
    it, nrelease, conv = 0, 0, false
    while (!conv && it < max_it)
        it += 1
        dr = -apply_reduced_hessian_inverse(H, g, active)
        if norm(dr, Inf) < sqrt(eps())
            # We have a solution candidate but need to check the gradient sign at the suggested active bounds. 
            # Find index of worst offender if it exists:
            rix =  get_index_worst_active_bound_candidate(u .+ d, g, active, lb, ub)
            if isnothing(rix)
                conv = true
            else
                # release index with largest violation from active candidate set
                nrelease += 1
                it -= 1 # don't count this as an iteration
                active[rix] = false
            end
        else
            ix, s = find_next_bounds(u .+ d, dr, active, lb, ub, active_chunk_tol)
            hits_bound = s <= 1.0
            s = min(1.0, s)
            sdr = s * dr
            if length(ix) > 0 && s > 0 && active_chunk_tol > 0
                # We may have slight (~tol) bound violations
                sdr = max.(lb, min.(ub, u .+ d .+ sdr)) .- (u .+ d)
            end
            # Update search direction
            d = d .+ sdr
            # Obtain gradient at u+d for quadratic model
            g = g0 .+ H * d
            # Add new active bound(s) guess for next iteration
            if hits_bound
                active[ix] .= true
            end
        end
    end
    qpinfo = (nits = it, conv = conv, nactive = count(active), nrelease = nrelease)
    # final projection just in case
    d = max.(lb, min.(ub, u .+ d)) .- u
    return (d, g, conv, qpinfo)
end

function get_index_worst_active_bound_candidate(u, g, active, lb, ub)
    # At active lower/upper bounds, gradient should be ≥ 0/≤ 0 respectively.
    # (corresponds to requiring positive lagrange multipliers in the KKT conditions)
    if !any(active)
        return nothing
    end
    ix = findall(active)
    mb = (lb[ix] .+ ub[ix]) ./ 2
    at_lower = u[ix] .< mb
    violations = zeros(Float64, length(ix))
    violations[at_lower] = max.(0, -g[ix][at_lower])  # Lower bound: gradient should be ≥ 0
    violations[.!at_lower] = max.(0, g[ix][.!at_lower])  # Upper bound: gradient should be ≤ 0
    val, idx = findmax(violations)
    if val < sqrt(eps())
        return nothing
    else
        return ix[idx]
    end
end

function solve_rough_qp(u0, g0, H, lb, ub, max_it)
    # Attempt to solve QP problem in a few iterations
    # Might not converge to optimal solution, but hopefully provides
    # good initial guess for solve_active_set_qp quickly
    active = get_active_bounds(u0, -g0, lb, ub)
    if all(active) || norm(g0[.!active], Inf) < sqrt(eps())
        qpinfo = (nits = 0, conv = true, nactive = count(active))
        return (zeros(size(u0)), g0, true, qpinfo)
    end
    u = copy(u0)
    g = copy(g0)
    it, conv = 0, false
    while (!conv && it < max_it)
        it = it + 1
        d = -apply_reduced_hessian_inverse(H, g, active)
        # new approximation
        u = max.(lb, min.(ub, u .+ d))
        # gradient at new point 
        g = g0 .+ H * (u .- u0)
        active = get_active_bounds(u, -g, lb, ub)
        conv = all(active) || norm(g[.!active], Inf) < sqrt(eps())
    end
    qpinfo = (nits = it, conv = conv, nactive = count(active))
    return (u .- u0, g, conv, qpinfo)
end

function proj_q(v, active)
    # project to null space of active bounds
    w = copy(v);
    w[active] .= 0.0
    return w
end

function get_active_bounds(u, v, lb, ub)
    # Determine active bounds based on current position u and search direction v
    tol = (ub .- lb) .* sqrt(eps())
    active_lower = (u .< lb .+ tol) .& (v .< 0)
    active_upper = (u .> ub .- tol) .& (v .> 0)
    active = active_upper .| active_lower
    return active
end

function find_next_bounds(u, d, active, lb, ub, tol)
    # Find next bounds that will be hit along direction d
    # filter out zero components (almost zero is fine since s -> ±Inf)
    dnz = copy(d)
    dnz[d .== 0] .= 1.0 
    sl = (lb .- u) ./ dnz
    su = (ub .- u) ./ dnz
    # pick whichever is positive (positive step along d)
    s = max.(sl, su) 
    # Disregard d = 0 / already active bounds
    s[active .| (d .== 0)] .= Inf
    # Find maximum step size before hitting next bound
    s_max, ix_min = findmin(s)
    # Check for other bounds hitting at same time (within tolerance)
    if s_max <= 1 && isfinite(s_max)
        ix = findall(s .<= (s_max + tol))
        # Select maximum (i.e., all become active/violated)
        if isempty(ix)
            println("s_max: ", s_max, "ix_min: ", ix_min)
        end
        s_max = maximum(s[ix])
    else
        ix = [ix_min]
    end
    return (ix, s_max)
end

function f_negative(u, f)
    # Negate objective and gradient for maximization
    v, g = f(u)
    return (-v, -g)
end

function f_scale(u, f, lb, ub)
    # Scale problem from [lb, ub] to [0, 1]
    v, g = f(u .* (ub .- lb) .+ lb)
    g = g .* (ub .- lb)
    return (v, g)
end

function handle_bounds!(lb, ub, n)
    if length(lb) == 1
        lb = fill(lb, n)
    elseif length(lb) != n
        error("Length of lower bound vector does not match number of variables.")
    end
    if length(ub) == 1
        ub = fill(ub, n)
    elseif length(ub) != n
        error("Length of upper bound vector does not match number of variables.")
    end
    @assert all(lb .< ub) "Lower bounds must be strictly less than or equal to upper bounds."
    return (lb, ub)
end

function incorporate_trust_region(u, r_trust, lb, ub)
    # Update bounds for current trust region radius (inf-norm)
    lb_cur = max.(lb, u .- r_trust)
    ub_cur = min.(ub, u .+ r_trust)
    return (lb_cur, ub_cur)
end

function update_trust_region!(r, rho, update, step, radius_increase, radius_decrease, ratio_thresholds)
    if rho < ratio_thresholds[1]
        r = radius_decrease * step * r
    elseif rho > ratio_thresholds[2] && r < update * step * (1 + sqrt(eps()))
        r = radius_increase * step * r
    end
    return r
end

function update_info!(info; obj_info = nothing, qp_info = nothing,
                     ls_info = nothing, tr_info = nothing
                    )
    if isnothing(obj_info)
        obj_info = (v = NaN, pg = NaN, n_active = 0)
    end
    if isnothing(qp_info)
        qp_info = (rough_solve = (nits = 0, conv = true, nactive = 0),
                   active_set  = (nits = 0, conv = true, nactive = 0, nrelease = 0),
                   success = true)
    end
    if isnothing(ls_info)
        ls_info  = ( flag = 1, step = NaN, nits = 0, objVals = [])
    end
    if isnothing(tr_info)
        tr_info  = (r_trust = NaN, rho = NaN)
    end
    
    info_step = (obj_info = obj_info, qp_info  = qp_info,
                 ls_info  = ls_info,  tr_info  = tr_info)
    if isnothing(info)
        info = [info_step]
    else
        info = push!(info, info_step)
    end
    return info
end

function print_info_step(info; it = length(info)-1)
    obj_info = info[it+1].obj_info
    qp_info  = info[it+1].qp_info
    ls_info  = info[it+1].ls_info
    tr_info  = info[it+1].tr_info
    
    @printf("It: %2d | v: %4.3e | ls-its: %2d | pg: %4.2e | ρ: %9.2e | qp-its: %2d +%3d | n-active: %3d\n",
            it, obj_info.v, isnan(ls_info.nits) ? 0 : Int(ls_info.nits),
            obj_info.pg, tr_info.rho, qp_info.rough_solve.nits, qp_info.active_set.nits , obj_info.n_active)
end

function print_end_message(stop_flags, stop_tols, info)
    @printf("\n*** Optimization stopped: ")
    if stop_flags[:maxit]
        @printf("maximum iterations (%d) reached. ***\n\n", stop_tols.maxit)
    elseif stop_flags[:grad]
        pg = info[end].obj_info.pg
        pg0 = info[1].obj_info.pg
        @printf("projected gradient norm %.2e < %.2e (relative %.2e < %.2e). ***\n\n", 
                pg, stop_tols.grad, pg/pg0, stop_tols.grad/pg0)
    elseif stop_flags[:obj_change]
        dobj = abs(info[end].obj_info.v - info[end-1].obj_info.v)
        obj0 = info[1].obj_info.v
        @printf("objective change %.2e < %.2e (relative %.2e < %.2e). ***\n\n", 
                dobj, stop_tols.obj_change, dobj/obj0, stop_tols.obj_change/obj0    )
    elseif stop_flags[:obj]
        obj0 = info[1].obj_info.v
        obj = info[end].obj_info.v
        @printf("objective value %.2e < %.2e (relative %.2e < %.2e). ***\n\n", 
                obj, stop_tols.obj, obj/obj0, stop_tols.obj/obj0)
    elseif stop_flags[:ls_fail]
        @printf("line search failed to find improvement in objective. ***\n\n")
    else
        @printf("??? unknown reason ??? *** \n\n")
    end
end

"""
Only for testing purposes with general (non-BFGS) Hessians 
"""
function apply_reduced_hessian_inverse(H, v, active)
    w = zeros(size(v))
    w[.!active] = H[.!active, .!active] \ v[.!active]
    return w
end

function apply_reduced_hessian(H, v, active)
    w = zeros(size(v))
    w[.!active] = H[.!active, .!active] * v[.!active]
    return w
end
