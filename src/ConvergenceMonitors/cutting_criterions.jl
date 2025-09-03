@kwdef mutable struct ConvergenceMonitorCuttingCriterion

    # Function to compute the distance from convergence
    distance_function = r -> compute_distance(r)
    # Count viloations per residual
    count_per_residual = false
    # Dict for storing contraction factor history
    history = nothing
    # Number of iterations to use for computing contraction factor metrics
    memory = 1
    # Target number of nonlinear iterations for the timestep
    target_iterations = 8
    # Max number of estimated iterations left for iterate to be classified as ok
    max_iterations_left = 4*target_iterations
    # Contraction factor parameters
    slow = 0.99
    fast = 0.1
    # Violation counter and limit for timestep cut
    num_violations::Vector{Int} = [0]
    num_violations_cut::Int = 3

end

"""
    set_convergence_monitor_cutting_criterion!(config; max_nonlinear_iterations = 50, kwargs...)

Utility for setting `ConvergenceMonitorCuttingCriterion` to the simulator
config. The function also adjusts the maximum number of nonlinear iterations (to
50 by default).
"""
function set_convergence_monitor_cutting_criterion!(config; max_nonlinear_iterations = 50, kwargs...)

    target_iterations = 8
    for sel in config[:timestep_selectors]
        if sel isa IterationTimestepSelector
            target_iterations = sel.target
            break
        end
    end
    cc = ConvergenceMonitorCuttingCriterion(; 
    target_iterations = target_iterations, kwargs...)
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
    N = max(cc.target_iterations - it + 1, 2)
    # First iterate number back in time to compute contraction factor
    it0 = max(it - cc.memory, 1)

    # Get distance from convergence
    report = step_reports[end][:errors]
    dist, names = cc.distance_function(report)
    # Reset cc if first iteration
    (it == 1) ? reset!(cc, dist, max_iter) : nothing
    # Store distance
    cc.history[:distance][it,:] .= dist

    # Compute contraction factor and iterations left to convergence
    θ, θ_target = compute_contraction_factor(cc.history[:distance][it0:it,:], N)
    its_left = iterations_left(θ, dist)
    # Check for oscillations
    oscillating_it = oscillation(cc.history[:distance][1:it,:])
    
    # Converged residuals should not be monitored
    converged = dist .== 0.0
    θ[converged] .= 0.0
    its_left[converged] .= 0
    oscillating_it[converged] .= false

    # println("Iterations left: $its_left")

    # Update history
    cc.history[:contraction_factor][it,:] .= θ
    cc.history[:contraction_factor_target][it,:] .= θ_target
    cc.history[:iterations_left][it,:] .= its_left
    cc.history[:oscillation][it,:] .= oscillating_it

    # Determine if current rate of convergence is adequate
    is_oscillating = vec(any(cc.history[:oscillation][it0:it,:], dims=1))
    good = θ .<= max.(θ_target, cc.fast) .&& .!is_oscillating
    good = good .|| converged
    ok = θ .<= cc.slow .&& its_left .<= cc.max_iterations_left
    bad = θ .> cc.slow .|| its_left .> cc.max_iterations_left

    if cc.count_per_residual
        cc.num_violations[good] .-= 1
        cc.num_violations[bad] .+= 1
    end

    if all(good)
        # Convergence rate good, decrease number of violations
        status = :good
        !cc.count_per_residual ? cc.num_violations .-= 1 : nothing
    elseif all(ok .|| good)
        # Convergence rate ok, keep number of violations
        status = :ok
    elseif any(bad)
        # Not converging, increase number of violations
        status = :bad
        !cc.count_per_residual ? cc.num_violations .+= 1 : nothing
    else
        # First iteration
        @assert it == 1
        status = :none
    end
    cc.history[:status][it] = status
    # Clamp number of violations to be non-negative
    cc.num_violations = max.(0, cc.num_violations)

    # Check if the number of violations exceeds the limit, in which case the
    # timstep should be aborted
    # early_cut = cc.num_violations > cc.num_violations_cut
    early_cut = any(cc.num_violations .> cc.num_violations_cut)
    
    # Generate convergence monitor report and store in step report
    cm_report = make_report(names, θ, θ_target, is_oscillating, status)
    step_reports[end][:convergence_monitor] = cm_report

    # Print status of convergence monitoring
    (cfg[:info_level] >= 2) ? print_convergence_status(cc, it, it0, names) : nothing

    return (relaxation, early_cut)

end

"""
    reset!(cc::ConvergenceMonitorCuttingCriterion, template, max_iter)

Utility for resetting convergence monitor cutting criterion.
"""
function reset!(cc::ConvergenceMonitorCuttingCriterion, template, max_iter)

    cc.num_violations = zeros(Int64, length(template))
    nc = max_iter + 1

    history = Dict()
    
    history[:distance] = Array{typeof(template[1])}(undef, nc, length(template))
    history[:contraction_factor] = Array{typeof(template[1])}(undef, nc, length(template))
    history[:contraction_factor_target] = Array{typeof(template[1])}(undef, nc, length(template))
    history[:iterations_left] = Array{Union{Int64, Float64}}(undef, nc, length(template))
    history[:status] = Vector{Symbol}(undef, nc)
    history[:oscillation] = Array{Bool}(undef, nc, length(template))

    history[:contraction_factor][1,:] .= NaN
    history[:contraction_factor_target][1,:] .= NaN
    history[:iterations_left][1] = NaN
    history[:status][1] = :none
    history[:oscillation][1] = false
    
    cc.history = history

end

"""
    make_report(θ, θ_target, oscillation, status)

Utility for generating a report from the convergence monitor.
"""
function make_report(names, θ, θ_target, oscillation, status)

    report = Dict()
    report[:names] = names
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
function print_convergence_status(cc::ConvergenceMonitorCuttingCriterion, it, it0, names)

    # Get convergence monitor status and actual and target contraction factor
    status = cc.history[:status][it]
    θ = cc.history[:contraction_factor][it,:]
    θ_target = cc.history[:contraction_factor_target][it,:]
    print_eq = length(θ) > 1
    # Find index of worst-offending residual
    Δθ = θ .- θ_target
    if status != :good
        Δθ[θ.<=cc.slow] = 0.0
    end
    _, worst_ix = findmax(Δθ)
    θ, θ_target = θ[worst_ix], θ_target[worst_ix]
    oscillation = any(cc.history[:oscillation][it0:it,:], dims=1)[worst_ix]
    # Find index of max estimated iterations left
    its_left, worst_ix_its = findmax(cc.history[:iterations_left][it,:])
    max_its = cc.max_iterations_left
    worst_ix = its_left > max_its ? worst_ix_its : worst_ix
    # Get convergence monitor 
    θ_slow = cc.slow
    θ_fast = cc.fast
    # Round values for printing
    round_local(x) = round(x; digits = 2)
    θ = round_local.(θ)
    θ_target = round_local.(θ_target)
    θ_slow = round_local.(θ_slow)
    θ_fast = round_local.(θ_fast)
    # Determine print message
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
            inequality = "θ_target = $θ_target < Θ = $θ ≤ $θ_slow = Θ_slow"
        end
        sym = " →"
        color = :yellow
    elseif status == :bad
        if its_left > max_its
            inequality = "Iterations left = $its_left > $max_its = upper limit"
        else
            inequality = "Θ = $θ > $θ_slow = Θ_slow"
        end
        color = :red
        sym = " ↑"
    else
        error("Unknown status: $status")
    end
    # Add reason and equation name
    if status != :none        
        reason = "\n\t\t Reason: " * inequality
        reason *= oscillation ? " (oscillation)" : ""
        reason *= print_eq ? ". Equation: $(names[worst_ix])" : ""
    end
    # Concatenate
    msg = "(It. $(it-1)): "
    msg *= "status = $(cc.history[:status][it]), "
    msg *= "violations = $(maximum((cc.num_violations)))" * sym
    msg *= reason
    msg *= "."
    # Print message
    Jutul.jutul_message("\tConvergence monitor", msg, color = color)

end