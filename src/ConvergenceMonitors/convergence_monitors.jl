function classify_iterate(distance, theta, theta_target, theta_slow, theta_fast, oscillation=missing)

    # Shorthand notation
    θ, θt, θs, θf = theta, theta_target, theta_slow, theta_fast
    osc = ismissing(oscillation) ? false : oscillation

    # Check convergence for all distances
    conv = distance .<= 1.0
    # Check contraction magnitude
    fast = θ .<= max.(θt, θf)
    ok = max.(θt, θf) .< θ .<= θs
    div = θ .> θs
    prt = false
    # Exclude "diverging" distances that have converged (e.g., θ = 1/1)
    div = div .&& .!conv
    # Classify fast contractions and converged distances as ok if oscillating
    ok = ok .|| (fast .|| conv) .&& osc
    # Converged, non-oscillating contractions are classifie as fast below
    ok = ok .&& .!(conv .&& .!osc)
    # Classify converged distances as fast, require no oscillation
    fast = (fast .|| conv) .&& .!osc
    # Sanity check
    @assert all(fast .+ ok .+ div .== 1)
    # Set status
    status = zeros(length(θ))
    status[fast] .= 1
    status[ok] .= 0
    status[div] .= -1
    # Return
    return status

end

function update_violation_count!(num_violations, status; strategy=:per_measure)

    fast = status .== 1
    diverge = status .== -1
    if strategy ∈ [:per_measure, :overall]
        num_violations[fast] .-= 1
        num_violations[diverge] .+= 1
    elseif strategy == :overall
        if all(fast)
            num_violations .-= 1
        elseif any(diverge)
            num_violations .+= 1
        end
    end
    clamp!(num_violations, 0, Inf)

    return num_violations

end

function analyze_step(distances, memory, target_its, theta_fast, theta_slow)

    # Shorthand notation
    θf, θs = theta_fast, theta_slow
    # Compute contraction factors and target contraction factors
    k = size(distances,1)
    if k == 1
        # Early return if we only have one iterate
        θ, θt = fill(NaN, size(distances,2)), fill(NaN, size(distances,2))
        osc = falses(size(distances,2))
        status = zeros(Int64, size(distances,2))
        return status, θ, θt, osc
    end
    N = max(target_its-k,2)
    δ = distances[max(k-memory,1):k, :]
    θ, θt = compute_contraction_factor(δ, N)
    # Check for oscillations
    osc = oscillation(distances[max(k-2,1):k,:])
    # Classify iterates
    status = classify_iterate(δ[end,:], θ, θt, θs, θf, osc)
    # Return
    return status, θ, θt, osc

end

function analyze_convergence(distances;
    memory=1, target_its=8, theta_fast=0.1, theta_slow=0.99)

    # Shorthand notation
    θf, θs = theta_fast, theta_slow
    # Preallocate
    Nv = zeros(Int64, size(distances))
    θ, θt = fill(NaN, size(distances)), fill(NaN, size(distances))
    osc = falses(size(distances))
    status = zeros(Int64, (size(distances)))
    # Loop over iterates
    Nv_k = copy(Nv[1,:])
    for k in 1:size(distances,1)
        status[k,:], θ[k,:], θt[k,:], osc[k,:] = 
        analyze_step(distances[1:k,:], memory, target_its, θf, θs)
        update_violation_count!(Nv_k, status[k,:]; strategy=:per_measure)
        Nv[k,:] .= Nv_k
        Nv_k = copy(Nv[k,:])
    end

    return status, Nv, θ, θt, osc

end

function analyze_convergence(step_reports::Vector; 
    pools = nothing, pooling_args = NamedTuple(), names = nothing, kwargs...)

    distances = Vector{Vector{Float64}}(undef, length(step_reports))
    for (k, step) in enumerate(step_reports)
        d, n = Jutul.ConvergenceMonitors.compute_distance(step[:errors];
        pools=pools, pooling_args=pooling_args)
        names = isnothing(names) ? n : names
        distances[k] = d
    end
    println(distances)
    distances = reduce(vcat, distances')
    return analyze_convergence(distances; kwargs...)

end

function analyze_convergence(timestep_report, ministep::Int; kwargs...)

    @assert 1 <= ministep <= length(timestep_report[:ministeps])
    @assert haskey(timestep_report[:ministeps][ministep], :steps)

    step_reports = timestep_report[:ministeps][ministep][:steps]
    return analyze_convergence(step_reports; kwargs...)

end

function analyze_convergence(step_reports::Vector, cc::ConvergenceMonitorCuttingCriterion)

    return analyze_convergence(step_reports;
    cc.distance_function_args...,
    memory=cc.memory, target_its=cc.target_iterations, 
    theta_fast=cc.fast, theta_slow=cc.slow)

end