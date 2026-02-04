@kwdef mutable struct ConvergenceMonitorCuttingCriterion

    # Function to compute the distance from convergence
    distance_function_args = NamedTuple()
    # Count viloations per residual measure, overall, or total
    strategy = :per_measure
    # Dict for storing contraction factor history
    history = nothing
    # Number of iterations to use for computing contraction factor metrics
    memory = 1
    # Target number of nonlinear iterations for the timestep
    target_iterations = 8
    # Max number of estimated iterations left for iterate to be classified as ok
    max_iterations_left = 2*target_iterations
    # Contraction factor parameters
    slow = 0.9
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

    # Get distance from convergence
    errors = step_reports[end][:errors]
    dist, names = compute_distance(errors; cc.distance_function_args...)
    # Reset cc if first iteration
    (it == 1) ? reset!(cc, dist, max_iter) : nothing
    # Store distances
    cc.history[:distance][it,:] .= dist
    # Analyze step
    status, θ, θ_target, osc = analyze_step(cc.history[:distance][1:it,:], cc)
    # Update violation count
    update_violation_count!(cc.num_violations, status; strategy=cc.strategy)
    # Update history
    cc.history[:contraction_factor][it,:] .= θ
    cc.history[:contraction_factor_target][it,:] .= θ_target
    cc.history[:oscillation][it,:] .= osc
    cc.history[:num_violations][it,:] .= cc.num_violations
    # Check if the number of violations exceeds the limit, in which case the
    # timestep should be aborted
    if cc.strategy ∈ [:per_measure, :overall]
        early_cut = any(cc.num_violations .> cc.num_violations_cut)
    elseif cc.strategy == :total
        early_cut = sum(cc.num_violations) > cc.num_violations_cut
    end
    # Generate convergence monitor report and store in step report
    cm_report = make_report(names, dist, θ, θ_target, osc, status)
    step_reports[end][:convergence_monitor] = cm_report
    # Print status of convergence monitoring
    (cfg[:info_level] >= 2) ? print_convergence_status(cc, it, names) : nothing

    return (relaxation, early_cut)

end


function analyze_step(distances, cc::ConvergenceMonitorCuttingCriterion)

    return analyze_step(distances, cc.memory, cc.target_iterations, cc.fast, cc.slow, cc.max_iterations_left)

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
    history[:oscillation] = Array{Bool}(undef, nc, length(template))
    history[:status] = Array{Int}(undef, nc, length(template))
    history[:num_violations] = Array{Int}(undef, nc, length(template))

    history[:contraction_factor][1,:] .= NaN
    history[:contraction_factor_target][1,:] .= NaN
    history[:oscillation][1] = false
    history[:status][1,:] .= 0
    history[:num_violations][1,:] .= 0
    
    cc.history = history

end

"""
    make_report(θ, θ_target, oscillation, status)

Utility for generating a report from the convergence monitor.
"""
function make_report(names, distance, θ, θ_target, oscillation, status)

    report = Dict()
    report[:names] = names
    report[:distance] = distance
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
function print_convergence_status(cc::ConvergenceMonitorCuttingCriterion, it, names)

    # Get convergence monitor status and actual and target contraction factor
    conv = cc.history[:distance][it,:] .<= 1.0
    θ = cc.history[:contraction_factor][it,:]
    θt = cc.history[:contraction_factor_target][it,:]
    θs = cc.slow
    θf = cc.fast
    Nv = cc.num_violations
    Nv0 = cc.history[:num_violations][max(it-1,1),:]
    # Find index of worst-offending residual
    Δθ = θ .- θt
    Δθ[conv] .= 0
    _, worst_ix = findmax(Δθ)
    θ, θt = θ[worst_ix], θt[worst_ix]
    osc = cc.history[:oscillation][it,:]
    θf = cc.fast
    # Round values for printing
    round_local(x) = round(x; digits = 2)
    θ_p = round_local.(θ)
    θt_p = round_local.(θt)
    θs_p = round_local.(cc.slow)
    θf_p = round_local.(cc.fast)
    θtf_p = max(θt_p, θf_p)

    # Make string for printing worst offending contraction factor
    worst = "Θ = $θ_p ∈ "
    if θ <= max(θt, θf)
        worst *= "(0.0, $θtf_p] = (0.0, max{Θ_target, θ_fast}] "
    elseif max(θt, θf) < θ <= θs
        worst *= "($θtf_p, $θs_p] = (max{Θ_target, θ_fast}, θ_slow] "
    elseif θs < θ
        worst *= "($θs_p, Inf) = (θ_slow, Inf) "
    else
        worst = "??? "
    end
    # Determine overall status
    if it == 1
        status_it = 2
    elseif cc.strategy ∈ [:per_measure, :overall]
        dnv = maximum(Nv) - maximum(Nv0)
        status_it = (dnv < 0) - (dnv > 0)
        status_it = maximum(Nv) == 0 ? 1 : status_it
    elseif cc.strategy .== :total
        dnv = sum(cc.num_violations[it,:]) - sum(cc.num_violations[it-1,:])
        status_it = (dnv < 0) - (dnv > 0) 
    end

    if status_it == 2
        worst, sym, color = "", "", :white
    elseif status_it == 1
        sym, color = " ↓", :green
    elseif status_it == 0
        sym, color = " →", :yellow
    elseif status_it == -1
        sym, color = " ↑", :red
    else
        error("Unknown status: $status_it")
    end
    # Add reason and equation name
    if !isnan(status_it)
        worst = "\n\t\t Worst: " * worst
        worst *= any(osc) ? "(oscillation) " : "" 
        worst *= "(Measure: $(names[worst_ix]))"
    end
    # Concatenate
    status_str = [:divergent, :acceptable, :good, :nothing]
    msg = "(It. $(it-1)): "
    msg *= "status = $(status_str[Int(status_it+2)]), "
    msg *= "violations = $(maximum((cc.num_violations)))" * sym
    msg *= worst
    msg *= "."
    # Print message
    Jutul.jutul_message("\tConvergence monitor", msg, color = color)

end