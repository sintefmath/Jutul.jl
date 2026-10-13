"""
    status = classify_iterate(distance, theta, theta_target, theta_slow, theta_fast; oscillation = false, N_left = missing)

Classify the current iterate for each measure as good (1), ok (0), or bad (-1).
With the lower and upper bounds θₗ = max(θ_target, θ_fast) and
θᵤ = max(θ_target, θ_slow), a measure is good if θ ≤ θₗ, ok if θₗ < θ ≤ θᵤ, and
bad if θ > θᵤ. The status is then adjusted in the following order:
converged measures (distance ≤ 1) are set to good, good measures that are
oscillating are lowered by one, and if `N_left = (N, N_max)` is given, measures
with θₗ < θ ≤ θᵤ and an estimated number of iterations left `N` exceeding
`N_max` are lowered by one.
"""
function classify_iterate(
        distance, theta, theta_target, theta_slow, theta_fast;
        oscillation = false, N_left = missing
    )

    # Shorthand notation
    θ, θt, θs, θf = theta, theta_target, theta_slow, theta_fast
    its_ok = ismissing(N_left) ? true : N_left[1] .<= N_left[2]

    # Check convergence for all distances
    conv = distance .<= 1.0
    # Check contraction magnitude
    θl = max.(θt, θf)
    θu = max.(θt, θs)
    fast = θ .<= θl
    ok = θl .< θ .<= θu
    slow = θ .> θu
    status = zeros(Int, length(θ))
    status[fast] .= 1
    status[ok] .= 0
    status[slow] .= -1
    # All converged measures should be at least be classified as ok
    status[conv] .= 1
    # Oscillating measures should be at most be classified as ok
    fast = status .== 1
    status[fast .&& oscillation] .-= 1
    # If approximated number of iterations left is larger than prescribed max,
    # the corresponding measure shold be classified as bad
    status[ok .&& .!its_ok] .-= 1

    @assert all(s -> -1 <= s <= 1, status)

    return status

end

"""
    update_violation_count!(num_violations, status; strategy = :per_measure)

Update the violation counter given the iterate status from
`classify_iterate`. With `strategy = :per_measure` or `:total`, the counter of
each measure is decreased by one for good and increased by one for bad
iterates. With `strategy = :overall`, all counters are decreased by one if all
measures are good and increased by one if any measure is bad. The counters are
bounded below by zero.
"""
function update_violation_count!(num_violations, status; strategy = :per_measure)

    fast = status .== 1
    slow = status .== -1
    if strategy in (:per_measure, :total)
        num_violations[fast] .-= 1
        num_violations[slow] .+= 1
    elseif strategy == :overall
        if all(fast)
            num_violations .-= 1
        elseif any(slow)
            num_violations .+= 1
        end
    else
        error("Unknown strategy: $strategy")
    end
    clamp!(num_violations, 0, Inf)

    return num_violations

end

"""
    Nv = count_violations(status; strategy = :per_measure)

Compute the history of the violation counter from a matrix of iterate statuses
(one row per iterate, one column per measure), see `update_violation_count!`.
"""
function count_violations(status; strategy = :per_measure)

    if strategy in (:per_measure, :total)
        Nv = zeros(Int64, size(status))
    else
        Nv = zeros(Int64, size(status, 1))
    end
    # Loop over iterates
    Nv_k = zeros(Int64, size(Nv, 2))
    for (k, sk) in enumerate(eachrow(status))
        update_violation_count!(Nv_k, sk; strategy = strategy)
        Nv[k, :] .= Nv_k
    end

    return Nv

end

"""
    status, θ, θ_target, osc = analyze_step(distances, memory, its_target, its_buffer, theta_fast, theta_slow, N_max; its_margin = 2)

Analyze the latest iterate given the distance history (one row per iterate, one
column per measure). The contraction factor is estimated from the last
`memory + 1` iterates, and the target contraction factor is computed so that
convergence is reached in max(its_target - j + 1 - its_margin, its_buffer)
iterations, where j is the zero-based index of the current iterate. The iterate
is then classified using `classify_iterate`, with `N_max` as the maximum
estimated number of iterations left for an ok iterate. For the first iterate,
θ and θ_target are NaN and the status is ok.
"""
function analyze_step(distances, memory, its_target, its_buffer, theta_fast, theta_slow, N_max; its_margin = 2)

    # Shorthand notation
    θf, θs = theta_fast, theta_slow
    # Compute contraction factors and target contraction factors
    k = size(distances, 1)
    if k == 1
        # Early return if we only have one iterate
        θ, θt = fill(NaN, size(distances, 2)), fill(NaN, size(distances, 2))
        osc = falses(size(distances, 2))
        status = zeros(Int64, size(distances, 2))
        return status, θ, θt, osc
    end
    # Remaining iterations to target: max(N - j + 1 - margin, buffer), where
    # j = k - 1 is the zero-based index of the current iterate. The default
    # margin of 2 gives max(N - k, buffer).
    N_target = max(its_target - (k - 1) + 1 - its_margin, its_buffer)
    δ = @view distances[max(k - memory, 1):k, :]
    θ, N, θt = compute_contraction_factor(δ, N_target)
    # Check for oscillations
    osc = oscillation(@view distances[max(k - 2, 1):k, :])
    # Classify iterates
    status = classify_iterate(
        δ[end, :], θ, θt, θs, θf;
        oscillation = osc, N_left = (N, N_max)
    )
    # Return
    return status, θ, θt, osc

end

"""
    cm = analyze_convergence(distance; memory = 1, target_its = 8, iteration_buffer = 2, iteration_margin = 2, max_iterations_left = 2*target_its, theta_fast = 0.1, theta_slow = 0.9, strategy = :per_measure)

Post-process a distance history (one row per iterate, one column per measure)
by applying `analyze_step` and `update_violation_count!` to each iterate, as
done by `ConvergenceMonitorCuttingCriterion` during simulation, with the same
defaults.
Returns a `Dict` with the distances, status, number of violations, contraction
factors, target contraction factors and oscillation flags for all iterates.
"""
function analyze_convergence(
        distance;
        memory = 1,
        target_its = 8,
        iteration_buffer = 2,
        iteration_margin = 2,
        max_iterations_left = 2 * target_its,
        theta_fast = 0.1,
        theta_slow = 0.9,
        strategy = :per_measure
    )

    # Shorthand notation
    θf, θs = theta_fast, theta_slow
    # Preallocate
    Nv = zeros(Int64, size(distance))
    θ, θt = fill(NaN, size(distance)), fill(NaN, size(distance))
    osc = falses(size(distance))
    status = zeros(Int64, (size(distance)))
    # Loop over iterates
    Nv_k = zeros(Int64, size(Nv, 2))
    for k in 1:size(distance, 1)
        status[k, :], θ[k, :], θt[k, :], osc[k, :] =
            analyze_step(
            view(distance, 1:k, :),
            memory,
            target_its,
            iteration_buffer,
            θf,
            θs,
            max_iterations_left;
            its_margin = iteration_margin
        )
        update_violation_count!(Nv_k, view(status, k, :); strategy = strategy)
        Nv[k, :] .= Nv_k
    end
    # Make convergence monitor dict
    cm = Dict()
    cm[:distance] = distance
    cm[:status] = status
    cm[:num_violations] = Nv
    cm[:contraction_factor] = θ
    cm[:contraction_factor_target] = θt
    cm[:oscillation] = osc

    return cm

end

"""
    cm = analyze_convergence(step_reports::Vector; pools = nothing, pooling_args = NamedTuple(), names = nothing, kwargs...)

Post-process the nonlinear iterations of a ministep. The distance from
convergence is computed for each step report using `compute_distance`, and the
result is analyzed using the matrix version of `analyze_convergence`, to which
`kwargs` are passed. The measure names are added to the returned `Dict`.
"""
function analyze_convergence(
        step_reports::Vector;
        pools = nothing, pooling_args = NamedTuple(), names = nothing, kwargs...
    )

    distance = Vector{Vector{Float64}}(undef, length(step_reports))
    for (k, step) in enumerate(step_reports)
        d, n = compute_distance(
            step[:errors];
            pools = pools, pooling_args = pooling_args
        )
        names = isnothing(names) ? n : names
        distance[k] = d
    end
    distance = reduce(vcat, distance')
    cm = analyze_convergence(distance; kwargs...)
    cm[:names] = names
    return cm

end

"""
    cm = analyze_convergence(timestep_report, ministep::Int; kwargs...)

Post-process the nonlinear iterations of ministep number `ministep` of a
timestep report.
"""
function analyze_convergence(timestep_report, ministep::Int; kwargs...)

    @assert 1 <= ministep <= length(timestep_report[:ministeps])
    @assert haskey(timestep_report[:ministeps][ministep], :steps)

    step_reports = timestep_report[:ministeps][ministep][:steps]
    return analyze_convergence(step_reports; kwargs...)

end
