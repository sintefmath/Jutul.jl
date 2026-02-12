export unit_box_bfgs
using Printf, SparseArrays, Polynomials
"""
    unit_box_bfgs(u0, f; kwargs...)

Iterative line search optimization using BFGS intended for scaled problems where 0 ≤ u ≤ 1 and f ~ O(1).

This is a port of the MRST function `unitBoxBFGS`, relicensed under MIT.

# Arguments
- `u0`: Initial guess vector of length n with 0 ≤ u0 ≤ 1, must be feasible with respect to additional constraints
- `f`: Function handle that returns tuple `(v, g)` where:
    * `v`: objective value
    * `g`: objective gradient vector of length n

# Keywords
- `maximize::Bool=false`: Set to true to maximize objective
- `step_init::Float64=NaN`: Initial step (gradient scaling). If NaN, uses `max_initial_update/max(|initial gradient|)`
- `time_limit::Float64=Inf`: Time limit for optimizer.
- `max_initial_update::Float64=0.05`: Maximum initial update step
- `history::Any=nothing`: For warm starting based on previous optimization (requires `output_hessian=true`)

## Stopping Criteria
- `grad_tol::Float64=1e-3`: Absolute tolerance of inf-norm of projected gradient
- `obj_change_tol::Float64=5e-4`: Absolute objective update tolerance
- `max_it::Int=25`: Maximum number of iterations

## Line Search Options
- `line_searchmax_it::Int=5`: Maximum number of line-search iterations
- `stepIncreaseTol::Float64=10`: Maximum step increase factor between line search iterations
- `wolfe1::Float64=1e-4`: Objective improvement condition
- `wolfe2::Float64=0.9`: Gradient reduction condition

## Hessian Approximation
- `use_bfgs::Bool=true`: Use BFGS (false for pure gradient search)
- `limited_memory::Bool=true`: Use L-BFGS instead of full Hessian approximations
- `lbfgs_num::Int=5`: Number of vector-pairs stored for L-BFGS
- `lbfgs_strategy::Symbol=:dynamic`: Strategy for L-BFGS (:static or :dynamic)

## Linear Constraints
- `lin_eq::NamedTuple{(:A,:b)}`: Linear equality constraints A*u=b
- `lin_ineq::NamedTuple{(:A,:b)}`: Additional linear inequality constraints A*u≤b
- `enforce_feasible::Bool=false`: Attempt to repair constraint violations by projection

## Display Options
- `plotEvolution::Bool=true`: Plot optimization progress
- `logPlot::Bool=false`: Use logarithmic y-axis for objective plotting
- `output_hessian::Bool=false`: Include Hessian approximation in history output

# Returns
- `v`: Optimal or best objective value
- `u`: Control/parameter vector corresponding to v
- `history`: Named tuple containing iteration history with fields:
    * `val`: objective values
    * `u`: control/parameter vectors
    * `pg`: projected gradient norms
    * `alpha`: line-search step lengths
    * `lsit`: number of line-search iterations
    * `lsfl`: line-search flags
    * `hess`: Hessian inverse approximations (if requested)
"""
function unit_box_bfgs(
        u0, f;
        maximize = false,
        step_init = NaN,
        max_initial_update = 0.05,
        grad_tol = 1.0e-3,
        obj_change_tol = 5.0e-4,
        obj_change_tol_rel = -Inf,
        max_it = 25,
        time_limit = Inf,
        use_bfgs = true,
        limited_memory = true,
        lbfgs_num = 5,
        lbfgs_strategy = :dynamic,
        lin_eq = NamedTuple(),
        lin_ineq = NamedTuple(),
        enforce_feasible = true,
        print = 1,
        output_hessian = false,
        history = nothing,
        kwarg...
    )
    t0 = time()
    if maximize
        f! = (u) -> f_as_negative(u, f)
        objSign = -1
    else
        f! = (u) -> f(u)
        objSign = 1
    end
    c = get_constraints(u0, lin_eq = lin_eq, lin_ineq = lin_ineq)
    if isnothing(history)
        # Setup constraint struct
        u0, flag, consOK = check_feasible(u0, c, enforce_feasible, nm = :Initial_Guess)
        @assert consOK "Infeasible initial guess"

        # Perform initial evaluation of objective and gradient:
        v0, g0 = f!(u0)
        # If not provided, set initial step
        step = step_init
        if isnan(step) || step <= 0
            step = max_initial_update / maximum(abs.(g0))
        end
        # Initial Hessian-approximation
        if limited_memory
            Hi = LimitedMemoryHessianLegacy(; init_scale = step, m = lbfgs_num, init_strategy = lbfgs_strategy)
        else
            Hi = step
        end
        HiPrev = deepcopy(Hi)
        it = 0
        # Setup struct for gathering optimization history
        history = update_history!([], objSign * v0, u0, norm(g0), NaN, NaN, NaN, Hi, output_hessian = output_hessian)
    else
        error("Warm-start not implemented yet")
    end
    v, u = deepcopy(v0), deepcopy(u0)
    if print > 0
        printInfo(history, it)
    end

    success = false
    g = deepcopy(g0)
    while !success
        it = it + 1
        d, Hi, pg, maxStep = get_search_direction(u0, g0, Hi, HiPrev, c)
        if !(norm(pg, Inf) < grad_tol) && !isempty(d)
            # Perform line-search
            tmp, flag, fixed = check_feasible(u0 + d, c, enforce_feasible)
            if !flag && fixed
                d = tmp - u0
            end
            u, v, g, lsinfo = line_search(u0, v0, g0, d, f!; maxStep = maxStep, kwarg...)
            # Update Hessian approximation
            if use_bfgs
                du, dg = u - u0, g - g0
                # if any of the gradient entries are not defined, we set
                # the difference to zero
                dg[.!isfinite.(dg)] .= 0
                if du' * dg > sqrt(eps()) * norm(du) * norm(dg)
                    HiPrev = deepcopy(Hi)
                    if isa(Hi, LimitedMemoryHessianLegacy)
                        Hi = update(Hi, du, dg)
                    else
                        r = 1 / (du' * dg)
                        nu = length(u)
                        V = sparse(1:nu, 1:nu, 1) - r * dg * du'
                        Hi = V' * Hi * V + r * (du * du')
                    end
                elseif it > 1
                    print_msg("Hessian not updated during iteration $it", :yellow)
                end
            end
            # Update history
            history = update_history!(
                history, objSign * v, u, norm(pg, Inf),
                lsinfo.step,
                lsinfo.nits,
                lsinfo.flag,
                Hi,
                output_hessian = output_hessian
            )
        else
            history = update_history!(
                history, objSign * v, u, norm(pg, Inf),
                0,
                0,
                0,
                Hi,
                output_hessian = output_hessian
            )
        end
        # Check stopping criteria t0
        grad_done = (norm(pg, Inf) < grad_tol)
        it_done = (it >= max_it)
        change_done = abs(v - v0) < obj_change_tol
        relchange_done = abs((v - v0) / v) < obj_change_tol_rel
        time_done = (time() - t0 > time_limit)
        # Either criterion is enough to terminate
        success = grad_done || it_done || change_done || relchange_done || time_done
        u0 = deepcopy(u)
        v0 = deepcopy(v)
        g0 = deepcopy(g)
        if print > 0
            printInfo(history, it)
        end
    end
    if maximize
        v = -v
    end
    return (v, u, history)
end

function box_bfgs(problem; kwarg...)
    ub = problem.limits.max
    lb = problem.limits.min
    return box_bfgs(problem.x0, problem, lb, ub; kwarg...)
end

function box_bfgs(x0, f, lb, ub; kwarg...)
    n = length(x0)
    length(lb) == n || throw(ArgumentError("Length of lower bound ($(length(lb))) must match length of initial guess ($n)"))
    length(ub) == n || throw(ArgumentError("Length of upper bound ($(length(ub))) must match length of initial guess ($n)"))
    # Check bounds
    for i in eachindex(x0, lb, ub)
        if lb[i] >= ub[i]
            throw(ArgumentError("Lower bound must be less than upper bound for index $i: lb[$i] = $(lb[i]), ub[$i] = $(ub[i])"))
        end
        if x0[i] < lb[i] || x0[i] > ub[i]
            throw(ArgumentError("Initial guess x0[$i] = $(x0[i]) is outside bounds [$(lb[i]), $(ub[i])]"))
        end
        if !isfinite(lb[i]) || !isfinite(ub[i])
            throw(ArgumentError("Bounds must be finite, got lb[$i] = $(lb[i]), ub[$i] = $(ub[i])"))
        end
    end
    δ = ub .- lb

    function dx_to_du!(g)
        for i in 1:n
            g[i] = g[i] * δ[i]
        end
    end

    function x_to_u(x)
        u = (x - lb) ./ δ
        return u
    end

    function u_to_x(u)
        x = u .* δ + lb
        return x
    end

    function F(u)
        x = u_to_x(u)
        obj, g = f(x)
        dx_to_du!(g)
        return (obj, g)
    end

    function transform_constraints(constraints)
        A_transformed = constraints.A .* reshape(δ, 1, :)
        b_transformed = constraints.b .- constraints.A * lb
        return (A = A_transformed, b = b_transformed)
    end

    # Transform linear constraints to unit box
    modified_kwarg = Dict()
    if haskey(kwarg, :lin_eq)
        modified_kwarg[:lin_eq] = transform_constraints(kwarg[:lin_eq])
    end
    if haskey(kwarg, :lin_ineq)
        modified_kwarg[:lin_ineq] = transform_constraints(kwarg[:lin_ineq])
    end
    modified_kwarg = merge(kwarg, modified_kwarg)

    u0 = x_to_u(x0)
    v, u, history = unit_box_bfgs(u0, F; modified_kwarg...)
    x = u_to_x(u)
    return (v, x, history)
end

function box_bfgs(u0, f, bounds; kwarg...)
    return box_bfgs(u0, f, bounds...; kwarg...)
end

function log_box_bfgs(x0, f, lb, ub; kwargs...)

    n = length(x0)

    length(lb) == n || throw(ArgumentError("Length of lower bound ($(length(lb))) must match length of initial guess ($n)"))
    length(ub) == n || throw(ArgumentError("Length of upper bound ($(length(ub))) must match length of initial guess ($n)"))

    # Check bounds and initial guess
    for i in eachindex(x0, lb, ub)
        if x0[i] ≤ 0 || lb[i] ≤ 0
            throw(ArgumentError("Log scaling requires positive values: x0[$i] = $(x0[i]), lb[$i] = $(lb[i]) must be > 0"))
        end
        if x0[i] < lb[i] || x0[i] > ub[i]
            throw(ArgumentError("Initial guess x0[$i] = $(x0[i]) is outside bounds [$(lb[i]), $(ub[i])]"))
        end
        if lb[i] >= ub[i]
            throw(ArgumentError("Lower bound must be less than upper bound for index $i: lb[$i] = $(lb[i]), ub[$i] = $(ub[i])"))
        end
    end

    # Log-transform bounds
    log_lb = log.(lb)
    log_ub = log.(ub)

    δ_log = log_ub .- log_lb

    # Transformation functions
    function x_to_u(x)
        log_x = log.(x)
        u = (log_x .- log_lb) ./ δ_log
        return u
    end

    function u_to_x(u)
        log_x = u .* δ_log .+ log_lb
        x = exp.(log_x)
        return x
    end

    function dx_to_du!(g, x)
        g .= g .* x .* δ_log  # Chain rule: df/du = df/dx * dx/du
    end

    # Wrapped objective function
    function F(u)
        x = u_to_x(u)
        obj, g = f(x)
        dx_to_du!(g, x)
        return (obj, g)
    end

    # Initial guess in transformed space
    u0 = x_to_u(x0)

    # Optimize in unit box
    v, u, history = unit_box_bfgs(u0, F; kwargs...)

    # Transform back to original space
    x = u_to_x(u)

    return (v, x, history)
end

function log_box_bfgs(u0, f, bounds; kwargs...)
    # Unpack bounds
    lb, ub = bounds
    return log_box_bfgs(u0, f, lb, ub; kwargs...)
end


function get_search_direction(u0, g0, Hi, HiPrev, c)
    # Find search-direction which is (sum of) the projection(s) of Hi*g0
    # restricted to directions with non-active constraints.
    cnt = 1
    for k in 1:3
        if k == 2
            Hi = deepcopy(HiPrev)
        elseif k == 3
            if isa(Hi, LimitedMemoryHessianLegacy)
                Hi = reset(Hi)
            else
                Hi = 1
            end
        end
        # Project gradient and search direction onto nullspace of equality constraints
        Q = deepcopy(c.e.Q)
        pg = -project_Q(g0, Q)
        d = -project_Q(g0, Q, Hi)
        # The following loop(s) should project onto nullspace of additional
        # active constraints. First for gradient direction, then for search
        # direction which may have additional constraints active. While-loop
        # will in most cases only be traversed once, so we don't worry too much
        # about efficiency
        isActive = false
        for kd in 1:2
            na, na_prev = 0, -1
            while na > na_prev
                if kd == 1
                    sgn, active_cur = classify_constraints(c.i.A, c.i.b, u0, pg)
                else
                    sgn, active_cur = classify_constraints(c.i.A, c.i.b, u0, d)
                end
                isActive = isActive .| active_cur
                na_prev = na
                na = sum(isActive)
                if na > na_prev
                    # redo projection for all active
                    Q, s, = svd(Array(hcat(c.i.A[isActive, :]', c.e.A'))) # large Matrix?
                    Q = Q[:, s .> sqrt(eps()) * s[1]]
                    if kd == 1
                        pg = -project_Q(g0, Q)
                    else
                        d = -project_Q(g0, Q, Hi)
                    end
                end
            end
        end
        # Check for tiny projected gradient
        if norm(pg, Inf) <= sqrt(eps()) * norm(g0, Inf)
            # nothing more to do
            d, maxStep = [], []
            return (d, Hi, pg, maxStep)
        end
        # Iteratively find all constraints that become active from u0 to u0+d,
        # and project remaining line segments accordingly.
        dr, gr = deepcopy(d), deepcopy(g0)
        becomesActive = deepcopy(isActive)
        d = 0.0
        done = false
        while !done
            if norm(dr) > sqrt(eps())
                sgn, = classify_constraints(c.i.A, c.i.b, u0 .+ d, dr)
                s, ix = find_next_constraint(c.i.A, c.i.b, u0 .+ d, dr, sgn .<= 0 .| becomesActive)
            else
                s, ix = 0.0, []
            end
            if (!isempty(ix)) && (s <= 1 + sqrt(eps()))
                becomesActive[ix] = true
                d = d .+ s * dr
                gr = (1 - s) * gr
                Q = expand_Q(Q, c.i.A[ix, :])
                dr = -project_Q(gr, Q, Hi)
            else
                d = d .+ dr
                done = true
            end
        end
        # find maximal step length we can take with d before hitting the next
        # constraint (should be >= 1)
        sgn, = classify_constraints(c.i.A, c.i.b, u0, d)
        maxStep, ix = find_next_constraint(c.i.A, c.i.b, u0, d, sgn .<= 0)
        if maxStep < 0.95
            print_msg("Problematic constraint handling, relative step length: $maxStep", :yellow)
        end
        if maxStep < 1
            d, maxStep = maxStep * d, 1
        end
        isDecreasing = d' * g0 <= 0
        isZero = norm(d, Inf) <= sqrt(eps()) * norm(Hi * g0, Inf)
        if isDecreasing && !isZero
            # decreasing search direction found, return
            return (d, Hi, pg, maxStep)
        else
            # retry with other Hessian approx
            if !isZero
                str = "Small norm of search direction"
            else
                str = "Non-inceasing search direction"
            end
            if cnt == 1
                print_msg("$str trying previous Hessian approximation.", :yellow)
            end
            if cnt == 2
                print_msg("$str trying to reset Hessian to identity.", :yellow)
            end
            if cnt == 3
                print_msg("Exiting: $str", :yellow)
                d, maxStep = [], []
            end
        end
    end
    return (d, Hi, pg, maxStep)
end

function project_Q(v, Q, H = nothing)
    if isnothing(H)
        H = 1
    else
        H = deepcopy(H)
    end
    if isempty(Q)
        w = H * v
    else
        if !isa(H, LimitedMemoryHessianLegacy)
            tmp = H * (v - Q * (Q' * v))
            w = tmp - Q * (Q' * tmp)
        else
            H = set_nullspace!(H, Q)
            w = H * v
        end
    end
    return vec(w)
end

function get_constraints(u; lin_eq = NamedTuple(), lin_ineq = NamedTuple())
    # Box constraints, always 0 <= u <= 1
    nu = length(u)
    A = [sparse(1:nu, 1:nu, -1.0); sparse(1:nu, 1:nu, 1.0)]
    b = [zeros(nu, 1); ones(nu, 1)]
    # Add general linear constraints
    if !isempty(lin_ineq)
        sc = opnorm(lin_ineq.A)
        A = [A; lin_ineq.A ./ sc]
        b = [b; lin_ineq.b ./ sc]
    end
    i = (A = A, b = b)
    # Equality constraints (always active)
    if !isempty(lin_eq)
        sc = opnorm(lin_eq.A)
        A = lin_eq.A ./ sc
        b = lin_eq.b ./ sc
        Q, s, = svd(Array(A'))
        Q = Q[:, s .> sqrt(eps()) * s[1]]
    else
        A = zeros(0, nu)
        b = zeros(0, 0)
        Q = zeros(nu, 0)
    end
    e = (A = A, b = b, Q = Q)
    return (i = i, e = e)
end

function check_feasible(u, c, enforce = false; nm = :Vector_u)
    # Check that u is feasible. If not and enforce == true, try to fix. Ideally
    # should be solved as ||u-u*|| st c.e and c.i, but since we don't have a
    # QP-solver resort to (costly) iterative projections in active subspaces.
    # Intended for fixing mild violations.
    hasEC = !isempty(c.e.A)
    hasIC = !isempty(c.i.A)
    ecOK = true
    icOK = true
    if hasEC
        Ae = c.e.A
        be = c.e.b
        if any(abs.(c.e.A * u - c.e.b) .> sqrt(eps()))
            # find closest u that fulfills c.e.
            u = u + Ae' * ((Ae * Ae') \ (be - Ae * u))
            ecOK = false # now OK, but warn
        end
    end
    flag = ecOK
    fixed = false
    max_it = 100
    It = 1
    for it in 1:max_it
        It = it
        if hasIC
            icOK = !any((c.i.A * u - c.i.b) .> sqrt(eps()))
            flag = flag & icOK
        end
        if !enforce
            break
        else
            if !icOK
                Q = zeros(length(u), 0)
                proj = (v) -> v
                if hasEC
                    Q = deepcopy(c.e.Q)
                    proj = (v) -> v - Q * (Q' * v)
                end
                # Loop through each violating and project. If there are no more
                # availabe directions, restart while-loop from current point
                done = false
                cnt = 0
                while !done
                    if cnt == 0
                        icIx = findall(vec(c.i.A * u - c.i.b) .> sqrt(eps()))
                        icIx = circshift(icIx, it)
                    end
                    if isempty(icIx) || cnt == length(icIx)
                        done = true
                    else
                        ix = icIx[cnt + 1]
                        a = c.i.A[ix, :] # column vector
                        b = c.i.b[ix]
                        pa = proj(a)
                        if norm(pa) < sqrt(eps()) * norm(a)
                            # skip
                            cnt = cnt + 1
                        else
                            cnt = 0
                            u = u + pa * ((b .- a' * u) / (a' * pa))
                            if size(Q, 2) < size(Q, 1) - 1
                                Q = expand_Q(Q, pa)
                                proj = (v) -> v - Q * (Q' * v)
                            else
                                # possibly restart while-loop
                                done = true
                            end
                        end
                    end
                end
            else
                fixed = true
                break
            end
        end
    end
    if It == max_it
        print_msg("Failed attempt to fix feasibility of $nm, continuing anyway ...", :yellow)
    elseif !flag
        if enforce
            u = Vector{eltype(u)}(u)
            print_msg("$nm was not feasible, fixed feasibility in $(It - 1) iteration(s)", :yellow)
        else
            print_msg("$nm is not feasible within tollerance. Consider running with option: enforce_feasible = true", :yellow)
        end
    end
    return (u, flag, fixed)
end

function classify_constraints(A, b, u, v)
    # classify inequality constraints for point u with direction v
    # sgn: -1: in, 0: parallell, 1: out
    # act: true for active
    sgn = A * v
    sgn[abs.(sgn) .< sqrt(eps())] .= 0
    sgn = sign.(sgn)
    act = (vec(A * u - b) .> -sqrt(eps())) .& (sgn .> 0) # BitVector
    return (sgn, act)
end

function find_next_constraint(A, b, u, d, ac)
    s = (b - A * u) ./ (A * d)
    s[ac] .= Inf
    s[s .< eps()] .= Inf
    return findmin(vec(s))
end

function expand_Q(Q, v)
    n0 = norm(v)
    v = v - Q * (Q' * v)
    if norm(v) / n0 > sqrt(eps())
        Q = hcat(Q, v / norm(v))
    else
        print_msg("Newly active constraint is linear combination of other active constraints ??!!", :yellow)
    end
    return Array(Q)
end

function f_as_negative(u, f)
    v, g = f(u)
    return (-v, -g)
end

mutable struct OptimizationHistory
    val
    u
    pg
    alpha
    lsit
    lsfl
    hess
    function OptimizationHistory(val = [], u = [], pg = [], alpha = [], lsit = [], lsfl = [], hess = [])
        return new(val, u, pg, alpha, lsit, lsfl, hess)
    end
end

Base.isempty(hst::OptimizationHistory) = false
function update_history!(hst::Union{OptimizationHistory, Vector}, val, u, pg, alpha, lsit, lsfl, hess; output_hessian = false)
    if !output_hessian
        hess = []
    end
    if isempty(hst)
        hst = OptimizationHistory()
    end
    push!(hst.val, val)
    push!(hst.u, u)
    push!(hst.pg, pg)
    push!(hst.alpha, alpha)
    push!(hst.lsit, lsit)
    push!(hst.lsfl, lsfl)
    hst.hess = (hst.hess..., hess)
    return hst
end

function printInfo(history, it)
    lsit = history.lsit[end]
    if it == 0
        println("It.  | Objective  | Proj. grad | Linesearch-its")
        println("-----------------------------------------------")
    end
    if isnan(lsit)
        lsit = "-"
    else
        lsit = "$(lsit)"
    end
    return @printf("%4d | %.4e | %.4e | %s\n", it, abs(history.val[end]), history.pg[end][end], lsit)
end

function line_search(
        u0, v0, g0, d, f;
        wolfe1 = 1.0e-4,
        wolfe2 = 0.9,
        safeguardFac = 1.0e-5,
        stepIncreaseTol = 10,
        line_searchmax_it = 5,
        maxStep = 1
    )
    c1, c2, sgf, incTol, max_it, aMax = wolfe1, wolfe2, safeguardFac, stepIncreaseTol, line_searchmax_it, maxStep

    # Assert search direction is increasing
    @assert d' * g0 <= 0
    # Define convenience-function to gather info for a "point", where a is the
    # step-length, v is the value and dv is the dirctional derivative:
    assign_point = (a, v, dv) -> (a = a, v = v, dv = dv)
    p0 = assign_point(0, v0, d' * g0) # 0-point
    # Function handles for wolfe conditions wrt p0
    w1 = (p) -> p.v <= p0.v + c1 * p.a * p0.dv
    w2 = (p) -> abs(p.dv) <= c2 * abs(p0.dv)
    # End-points of initial interval
    p1 = p0
    p2 = assign_point(aMax, Inf, Inf)
    # Initial step-length:
    a = 1
    line_search_done = false
    it = 0
    vals = fill(NaN, (1, max_it))
    while !line_search_done && it < max_it
        it = it + 1
        u = u0 + a * d
        v, g = f(u)
        vals[it] = v
        p = assign_point(a, v, d' * g)
        if w1(p) && w2(p)
            line_search_done, flag = true, 1
        else
            if (abs(aMax - p.a) < sqrt(eps())) && # max step-length reached
                    (p.v < p0.v) && # the step yielded an improvement
                    (p.dv < 0)  # continuing further would improve (but not allowed)
                line_search_done, flag = true, -1
                print_msg("Line search at max step size, Wolfe conditions not satisfied for this step", :red)
            else
                # logic for refining/expanding interval of interest [p1 p2]
                if p.a > p2.a
                    # if we have extrapolated p1.v >= p2.v
                    p1, p2 = p2, p
                else
                    if p.dv >= 0
                        p2 = p
                    else
                        if p1.v <= p2.v
                            p2 = p
                        else
                            p1 = p
                        end
                    end
                end
                # Find next candidate-step by interpolation
                if p1.v > p2.v && p1.dv >= p2.dv
                    # Just getting steeper, no need to interpolate
                    a = Inf
                else
                    a = argmax_cubic(negate_point(p1), negate_point(p2))
                end
                # Safe-guarding and thresholding:
                if a > p2.a
                    a = max(a, (1 + sgf) * p2.a)
                    a = min(a, min(incTol * p2.a, aMax))
                elseif a > p1.a
                    a = max(a, p1.a + sgf * (p2.a - p1.a))
                    a = min(a, p2.a - sgf * (p2.a - p1.a))
                else
                    a = (p1.a + p2.a) / 2
                    print_msg("Cubic interpolation failed, cutting interval in half ...", :yellow)
                end
            end
        end
    end
    # Check if line search succeeded
    if ! line_search_done
        flag = -2
        print_msg("Line search unable to succeed in $max_it iterations ...", :yellow)
        # Although line search did not succeed in max_it iterations, we ensure
        # to return the greater of p1 and p2's objective value none the less.
        if p1.v < p2.v
            u = u0 + p1.a * d
            # and we will re-compute the gradients for these controls
            v, g = f(u)
        end
    end
    info = (flag = flag, step = a, nits = it, objVals = vals[1:it])
    return (u, v, g, info)
end

function negate_point(p)
    p_neg = merge(deepcopy(p), (v = -p.v, dv = -p.dv))
    return p_neg
end

function argmax_cubic(p1::NamedTuple, p2::NamedTuple)
    # find max of cubic polynomial through p1, p2
    shift = p1.a
    p1 = merge(p1, (a = 0,))
    p2 = merge(p2, (a = p2.a - shift,))
    a = p2.a
    c = zeros(4)
    # Attention!!
    # Both the coefficients of polynomial and its derivatives are in reverse order in contrast to Matlab
    # Indexing of polynomial starts at 0 (hence poly = poly[0] + ... + poly[3]*x^3)
    c[4:-1:3] .= [a^3 a^2; 3 * a^2 2 * a] \ [p2.v - p1.dv * a - p1.v; p2.dv - p1.dv]
    c[2:-1:1] .= [p1.dv; p1.v]
    poly = Polynomial(c)
    xe = roots(derivative(poly))
    if length(xe) == 0
        xe = -Inf
    elseif any(imag(xe) .!= 0)
        xe = Inf
    elseif xe[1] == xe[end]
        if poly[3] != 0
            xe = Inf
        elseif poly[2] < 0
            xe = xe[1]
        else
            xe = -Inf
        end
    else
        f, ix = findmax(poly.(xe))
        xe = xe[ix]
        if xe < p1.a
            xe = -Inf
        end
    end
    return xe + shift
end

function print_msg(msg, color)
    Jutul.jutul_message("LBFGS", msg, color = color)
end
