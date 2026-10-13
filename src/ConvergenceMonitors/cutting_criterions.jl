"""
    ConvergenceMonitorCuttingCriterion(; kwargs...)

Cutting criterion that monitors the convergence of the nonlinear iterations and
aborts the timestep if convergence is too slow. See
`Jutul.cutting_criterion(::ConvergenceMonitorCuttingCriterion, ...)` for
details. Usually set up with `set_convergence_monitor_cutting_criterion!`.

# Keyword arguments
- `distance_function_args = NamedTuple()`: keyword arguments passed to `compute_distance`.
- `strategy = :per_measure`: count violations per measure (`:per_measure`), for all measures at once (`:overall`), or per measure with the sum compared to the limit (`:total`).
- `memory = 1`: number of previous iterates used to estimate the contraction factor.
- `iteration_target = 8`: target number of nonlinear iterations for the timestep.
- `iteration_buffer = 2`: lower bound on the number of remaining iterations used for the target contraction factor.
- `iteration_margin = 2`: margin subtracted from the remaining iterations, max(N - k + 1 - margin, buffer).
- `max_iterations_left = 2*iteration_target`: maximum estimated number of iterations left for an iterate to be classified as ok.
- `slow = 0.9`, `fast = 0.1`: bounds on the contraction factor for slow and fast convergence.
- `num_violations_cut = 3`: the timestep is aborted if the number of violations exceeds this limit.
"""
@kwdef mutable struct ConvergenceMonitorCuttingCriterion

    # Keyword arguments to compute the distance from convergence
    distance_function_args = NamedTuple()
    # Count violations per residual measure, overall, or total
    strategy = :per_measure
    # Dict for storing convergence monitor history for the current ministep
    history = nothing
    # Number of iterations to use for computing contraction factor metrics
    memory = 1
    # Target number of nonlinear iterations for the timestep
    iteration_target = 8
    # Lower bound on the number of remaining iterations used for the target
    # contraction factor
    iteration_buffer = 2
    # Margin subtracted from the remaining iterations, N - k + 1 - margin
    iteration_margin = 2
    # Max number of estimated iterations left for iterate to be classified as ok
    max_iterations_left = 2 * iteration_target
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
config. The iteration target is taken from the first `IterationTimestepSelector`
in the config (8 if there is none), and `kwargs` are passed to the
`ConvergenceMonitorCuttingCriterion` constructor. The function also sets the
maximum number of nonlinear iterations (to 50 by default).
"""
function set_convergence_monitor_cutting_criterion!(config; max_nonlinear_iterations = 50, kwargs...)

    iteration_target = 8
    for sel in config[:timestep_selectors]
        if sel isa IterationTimestepSelector
            iteration_target = sel.target
            break
        end
    end
    cc = ConvergenceMonitorCuttingCriterion(;
        iteration_target = iteration_target, kwargs...
    )
    config[:cutting_criterion] = cc
    config[:max_nonlinear_iterations] = max_nonlinear_iterations
    return config

end

"""
    Jutul.cutting_criterion(cc::ConvergenceMonitorCuttingCriterion, sim, dt, forces, it, max_iter, cfg, e, step_reports, relaxation)

Cutting criterion based on monitoring convergence. The function computes the
distance to convergence (see `compute_distance`) and estimates the contraction
factor from the recent iterates, assuming that they follow a geometric series.
It also computes the target contraction factor needed to converge within the
iteration target, and checks if the iterates are oscillating. Based on this,
the iterate is classified as good, ok, or bad (see `analyze_step`), and the
violation counter is updated accordingly (+1 for bad, -1 for good). The
timestep is aborted if the number of violations exceeds `num_violations_cut`.
The convergence monitor report is stored in the step report under
`:convergence_monitor`.
"""
function Jutul.cutting_criterion(cc::ConvergenceMonitorCuttingCriterion, sim, dt, forces, it, max_iter, cfg, e, step_reports, relaxation)

    # Get distance from convergence
    errors = step_reports[end][:errors]
    dist, names = compute_distance(errors; cc.distance_function_args...)
    # Reset cc if first iteration
    if it == 1
        reset!(cc, dist, max_iter)
    end
    # Store distances
    cc.history[:distance][it, :] .= dist
    # Analyze step
    status, θ, θ_target, osc = analyze_step(view(cc.history[:distance], 1:it, :), cc)
    # Update violation count
    update_violation_count!(cc.num_violations, status; strategy = cc.strategy)
    # Update history
    cc.history[:contraction_factor][it, :] .= θ
    cc.history[:contraction_factor_target][it, :] .= θ_target
    cc.history[:oscillation][it, :] .= osc
    cc.history[:status][it, :] .= status
    cc.history[:num_violations][it, :] .= cc.num_violations
    # Check if the number of violations exceeds the limit, in which case the
    # timestep should be aborted
    if cc.strategy in (:per_measure, :overall)
        early_cut = any(cc.num_violations .> cc.num_violations_cut)
    elseif cc.strategy == :total
        early_cut = sum(cc.num_violations) > cc.num_violations_cut
    else
        error("Unknown strategy: $(cc.strategy)")
    end
    # Generate convergence monitor report and store in step report
    cm_report = make_report(names, dist, θ, θ_target, osc, status)
    step_reports[end][:convergence_monitor] = cm_report
    # Print status of convergence monitoring
    if cfg[:info_level] >= 2
        print_convergence_status(cc, it, names)
    end

    return (relaxation, early_cut)

end

function analyze_step(distances, cc::ConvergenceMonitorCuttingCriterion)

    return analyze_step(distances, cc.memory, cc.iteration_target, cc.iteration_buffer, cc.fast, cc.slow, cc.max_iterations_left; its_margin = cc.iteration_margin)

end

"""
    reset!(cc::ConvergenceMonitorCuttingCriterion, template, max_iter)

Reset the violation counter and allocate the history of the convergence monitor
cutting criterion for a new ministep with at most `max_iter + 1` iterates and
the same number of measures as `template`.
"""
function reset!(cc::ConvergenceMonitorCuttingCriterion, template, max_iter)

    n = length(template)
    T = eltype(template)
    cc.num_violations = zeros(Int64, n)
    nc = max_iter + 1

    history = Dict()

    history[:distance] = Array{T}(undef, nc, n)
    history[:contraction_factor] = Array{T}(undef, nc, n)
    history[:contraction_factor_target] = Array{T}(undef, nc, n)
    history[:oscillation] = Array{Bool}(undef, nc, n)
    history[:status] = Array{Int}(undef, nc, n)
    history[:num_violations] = Array{Int}(undef, nc, n)

    history[:contraction_factor][1, :] .= NaN
    history[:contraction_factor_target][1, :] .= NaN
    history[:oscillation][1, :] .= false
    history[:status][1, :] .= 0
    history[:num_violations][1, :] .= 0

    cc.history = history
    return cc

end

"""
    make_report(names, distance, θ, θ_target, oscillation, status)

Utility for generating a convergence monitor report for the current iterate.
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
    print_convergence_status(cc::ConvergenceMonitorCuttingCriterion, it, names)

Utility for printing the status of the convergence monitor, including the
measure whose contraction factor exceeds its target the most.
"""
function print_convergence_status(cc::ConvergenceMonitorCuttingCriterion, it, names)

    # Get convergence monitor status and actual and target contraction factor
    conv = cc.history[:distance][it, :] .<= 1.0
    θ = cc.history[:contraction_factor][it, :]
    θt = cc.history[:contraction_factor_target][it, :]
    θs = cc.slow
    θf = cc.fast
    Nv = cc.num_violations
    Nv0 = cc.history[:num_violations][max(it - 1, 1), :]
    # Find index of worst-offending residual
    Δθ = θ .- θt
    Δθ[conv] .= 0
    _, worst_ix = findmax(Δθ)
    θ, θt = θ[worst_ix], θt[worst_ix]
    osc = cc.history[:oscillation][it, :]
    # Round values for printing
    round_local(x) = round(x; digits = 2)
    θ_p = round_local.(θ)
    θt_p = round_local.(θt)
    θs_p = round_local(θs)
    θf_p = round_local(θf)
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
    elseif cc.strategy in (:per_measure, :overall)
        dnv = maximum(Nv) - maximum(Nv0)
        status_it = (dnv < 0) - (dnv > 0)
        status_it = maximum(Nv) == 0 ? 1 : status_it
    elseif cc.strategy == :total
        dnv = sum(Nv) - sum(Nv0)
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
    if status_it != 2
        worst = "\n\t\t Worst: " * worst
        worst *= any(osc) ? "(oscillation) " : ""
        worst *= "(Measure: $(names[worst_ix]))"
    end
    # Concatenate
    status_str = [:divergent, :acceptable, :good, :nothing]
    msg = "(It. $(it - 1)): "
    msg *= "status = $(status_str[Int(status_it + 2)]), "
    nv = cc.strategy == :total ? sum(Nv) : maximum(Nv)
    msg *= "violations = $nv" * sym
    msg *= worst
    msg *= "."
    # Print message
    return Jutul.jutul_message("\tConvergence monitor", msg, color = color)

end
