@kwdef mutable struct ConvergenceMonitorCuttingCriterion

    # Function to compute the distance from convergence
    distance_function = r -> compute_distance(r)
    # Dict for storing contraction factor history
    history = nothing
    # Number of iterations to use for computing contraction factor metrics
    memory = 1
    # Target number of nonlinear iterations for the timestep
    target_iterations = 5
    # Contraction factor parameters
    slow = 0.99
    fast = 0.1
    # Violation counter and limit for timestep cut
    num_violations::Int     = 0
    num_violations_cut::Int = 3

end

"""
    set_convergence_monitor_cutting_criterion!(config; max_nonlinear_iterations = 50, kwargs...)

Utility for setting `ConvergenceMonitorCuttingCriterion` to the simulator
config. The function also adjusts the maximum number of nonlinear iterations (to
50 by default).
"""
function set_convergence_monitor_cutting_criterion!(config; max_nonlinear_iterations = 50, kwargs...)

    cc = ConvergenceMonitorCuttingCriterion(; kwargs...)
    config[:cutting_criterion] = cc
    config[:max_nonlinear_iterations] = max_nonlinear_iterations

end

"""
    Jutul.cutting_criterion(cc::ConvergenceMonitorCuttingCriterion, sim, dt, forces, it, max_iter, cfg, e, step_reports, relaxation)

Cutting ctriterion based on monitoring convergence. The function computes the
contraction factor from the distance to convergence (in a user-defined metric)
between consecutive iterates. The function also computes the target contraction
factor under the assumption that the iterates follow a geometric series, and
checks if the iterates are oscillating. Based on this, the iterate is classified
as "good", "ok", or "bad", and a counter for number of violations is updated
accordingly (+1 for "bad", -1 for "good"). The timestep is aborted if the number
of violations exceeds a user-defined limit.
"""
function Jutul.cutting_criterion(cc::ConvergenceMonitorCuttingCriterion, sim, dt, forces, it, max_iter, cfg, e, step_reports, relaxation)
    
    # Maximum number of iterations left if we should converge in
    # target_iterations iterations
    N = max(max_iter - it + 1, 2)
    # First iterate number back in time to compute contraction factor
    it0 = max(it - cc.memory, 1)

    # Get distance from convergence
    report = step_reports[end][:errors]
    dist, = cc.distance_function(report)
    # Reset cc if first iteration
    (it == 1) ? reset!(cc, dist, max_iter) : nothing
    # Store distance
    cc.history[:distance][it] = dist

    # Compute contraction factor and update history
    Θ, Θ_target = compute_contraction_factor(cc.history[:distance][it0:it], N)
    cc.history[:contraction_factor][it] = Θ
    cc.history[:contraction_factor_target][it] = Θ_target
    # Check if the contraction factors are oscillating
    oscillating_it = oscillation(cc.history[:contraction_factor][1:it], cc.slow)
    cc.history[:oscillation][it] = oscillating_it
    is_oscillating = any(cc.history[:oscillation][it0:it])
    
    # Determine if current rate of convergence is adequate
    good = all(Θ .<= max(Θ_target, cc.fast)) && !is_oscillating
    ok = all(Θ .<= cc.slow)
    bad = any(Θ .> cc.slow)

    if good
        # Convergence rate good, decrease number of violations
        cc.num_violations -= 1
        status = :good
    elseif ok
        # Convergence rate ok, keep number of violations
        status = :ok
    elseif bad
        # Not converging, increase number of violations
        cc.num_violations += 1
        status = :bad
    else
        # First iteration
        @assert it == 1
        status = :none
    end
    cc.history[:status][it] = status
    # Clamp number of violations to be non-negative
    cc.num_violations = max(0, cc.num_violations)

    # Check if the number of violations exceeds the limit, in which case the
    # timstep should be aborted
    early_cut = cc.num_violations > cc.num_violations_cut
    
    # Generate convergence monitor report and store in step report
    cm_report = make_report(Θ, Θ_target, is_oscillating, status)
    step_reports[end][:convergence_monitor] = cm_report

    # Print status of convergence monitoring
    (cfg[:info_level] >= 2) ? print_convergence_status(cc, it, it0) : nothing

    return (relaxation, early_cut)

end

"""
    reset!(cc::ConvergenceMonitorCuttingCriterion, template, max_iter)

Utility for resetting convergence monitor cutting criterion.
"""
function reset!(cc::ConvergenceMonitorCuttingCriterion, template, max_iter)

    cc.num_violations = 0
    nc = max_iter + 1

    history = Dict()
    
    history[:distance] = Array{typeof(template[1])}(undef, nc, length(template))
    history[:contraction_factor] = Array{typeof(template[1])}(undef, nc, length(template))
    history[:contraction_factor_target] = Array{typeof(template[1])}(undef, nc, length(template))
    history[:status] = Array{Symbol}(undef, nc, length(template))
    history[:oscillation] = Array{Bool}(undef, nc, length(template))

    history[:contraction_factor][1] = NaN
    history[:contraction_factor_target][1] = NaN
    history[:status][1] = :none
    history[:oscillation][1] = false
    
    cc.history = history

end

"""
    make_report(θ, θ_target, oscillation, status)

Utility for generating a report from the convergence monitor.
"""
function make_report(θ, θ_target, oscillation, status)

    report = Dict()
    report[:contraction_factor] = θ
    report[:contraction_factor_target] = θ_target
    report[:oscillation] = oscillation
    report[:status] = status
    return report

end

"""
    print_convergence_status(cc::ConvergenceMonitorCuttingCriterion, it, it0)

Utility for printing the status of the convergence monitor.
"""
function print_convergence_status(cc::ConvergenceMonitorCuttingCriterion, it, it0)

    round_local(x) = round(x; digits = 2)

    θ = cc.history[:contraction_factor][it]
    θ_target = cc.history[:contraction_factor_target][it]
    θ_slow = cc.slow
    θ_fast = cc.fast
    status = cc.history[:status][it]
    oscillation = any(cc.history[:oscillation][it0:it])

    θ = round_local(θ)
    θ_target = round_local(θ_target)
    θ_slow = round_local(θ_slow)
    θ_fast = round_local(θ_fast)
    
    if status == :none
        inequality, sym, color, reason = "", "", :white, ""
    elseif status == :good
        inequality = "Θ = $θ ≤ max{$θ_target, $θ_fast} = max{Θ_target, θ_fast}"
        sym = " ↓"
        color = :green
    elseif status == :ok
        if θ < max(θ_target, θ_fast)
            inequality = "Θ = $θ ≤ max{$θ_target, $θ_fast} = max{Θ_target, θ_fast}"
        else
            inequality = "Θ = $θ ≤ $θ_slow = Θ_slow"
        end
        sym = " →"
        color = :yellow
    elseif status == :bad
        inequality = "Θ = $θ > $θ_slow = Θ_slow"
        color = :red
        sym = " ↑"
    else
        error("Unknown status: $status")
    end

    if status != :none        
        reason = "\n\t\t Reason: " * inequality
        reason *= oscillation ? " (oscillation)" : ""
    end

    msg = "(It. $(it-1)): "
    msg *= "status = $(cc.history[:status][it]), "
    msg *= "violations = $(cc.num_violations)" * sym
    msg *= reason
    msg *= "."
    Jutul.jutul_message("\tConvergence monitor", msg, color = color)

end