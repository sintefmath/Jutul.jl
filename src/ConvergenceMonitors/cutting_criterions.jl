@kwdef mutable struct ContractionFactorCuttingCriterion

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

function set_contraction_factor_cutting_criterion!(config; max_nonlinear_iterations = 50, kwargs...)

    cc = ContractionFactorCuttingCriterion(; kwargs...)
    config[:cutting_criterion] = cc
    config[:max_nonlinear_iterations] = max_nonlinear_iterations

end

function Jutul.cutting_criterion(cc::ContractionFactorCuttingCriterion, sim, dt, forces, it, max_iter, cfg, e, step_reports, relaxation)
    
    N = max(max_iter - it + 1, 2)
    it0 = max(it - cc.memory, 1)

    report = step_reports[end][:errors]
    dist, = cc.distance_function(report)

    (it == 1) ? reset!(cc, dist, max_iter) : nothing

    # Store distance
    cc.history[:distance][it] = dist

    Θ, Θ_target = compute_contraction_factor(cc.history[:distance][it0:it], N)
    cc.history[:contraction_factor][it] = Θ
    cc.history[:contraction_factor_target][it] = Θ_target

    oscillating_it = oscillation(cc.history[:contraction_factor][1:it], cc.slow)
    cc.history[:oscillation][it] = oscillating_it
    is_oscillating = any(cc.history[:oscillation][it0:it])
    
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
    cc.num_violations = max(0, cc.num_violations)
    cc.history[:status][it] = status

    early_cut = cc.num_violations > cc.num_violations_cut
    step_reports[end][:cutting_criterion] = cc

    if cfg[:info_level] >= 2
        print_progress(cc, it, it0)
    end

    return (relaxation, early_cut)

end

function reset!(cc::ContractionFactorCuttingCriterion, template, max_iter)

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

function print_progress(cc::ContractionFactorCuttingCriterion, it, it0)

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