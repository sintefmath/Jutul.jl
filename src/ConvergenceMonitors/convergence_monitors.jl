function classify_iterate(distance, theta, theta_target, theta_slow, theta_fast, oscillation=missing)

    # Shorthand notation
    θ, θt, θs, θf = theta, theta_target, theta_slow, theta_fast
    osc = ismissing(oscillation) ? false : oscillation

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
    status[fast .&& osc] .-= 1
    
    @assert all([s ∈ (-1:1) for s in status])

    return status

end

function update_violation_count!(num_violations, status; strategy=:per_measure)

    fast = status .== 1
    slow = status .== -1
    if strategy ∈ [:per_measure, :total]
        num_violations[fast] .-= 1
        num_violations[slow] .+= 1
    elseif strategy == :overall
        if all(fast)
            num_violations .-= 1
        elseif any(slow)
            num_violations .+= 1
        end
    end
    clamp!(num_violations, 0, Inf)

    return num_violations

end

function count_violations(status; strategy=:per_measure)
    
    if strategy ∈ [:per_measure, :total]
        Nv = zeros(Int64, size(status))
    else
        Nv = zeros(Int64, size(status,1))
    end
    # Loop over iterates
    Nv_k = copy(Nv[1,:])
    for (k, sk) in enumerate(eachrow(status))
        update_violation_count!(Nv_k, sk; strategy=strategy)
        Nv[k,:] .= Nv_k
        Nv_k = copy(Nv[k,:])
    end

    return Nv

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

function analyze_convergence(distance;
    memory=1, target_its=8, theta_fast=0.1, theta_slow=0.99, strategy=:per_measure)

    # Shorthand notation
    θf, θs = theta_fast, theta_slow
    # Preallocate
    Nv = zeros(Int64, size(distance))
    θ, θt = fill(NaN, size(distance)), fill(NaN, size(distance))
    osc = falses(size(distance))
    status = zeros(Int64, (size(distance)))
    # Loop over iterates
    Nv_k = copy(Nv[1,:])
    for k in 1:size(distance,1)
        status[k,:], θ[k,:], θt[k,:], osc[k,:] = 
        analyze_step(distance[1:k,:], memory, target_its, θf, θs)
        update_violation_count!(Nv_k, status[k,:]; strategy=strategy)
        Nv[k,:] .= Nv_k
        Nv_k = copy(Nv[k,:])
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

function analyze_convergence(step_reports::Vector; 
    pools = nothing, pooling_args = NamedTuple(), names = nothing, kwargs...)

    distance = Vector{Vector{Float64}}(undef, length(step_reports))
    for (k, step) in enumerate(step_reports)
        d, n = Jutul.ConvergenceMonitors.compute_distance(step[:errors];
        pools=pools, pooling_args=pooling_args)
        names = isnothing(names) ? n : names
        distance[k] = d
    end
    distance = reduce(vcat, distance')
    cm = analyze_convergence(distance; kwargs...)
    cm[:names] = names
    return cm

end

function analyze_convergence(timestep_report, ministep::Int; kwargs...)

    @assert 1 <= ministep <= length(timestep_report[:ministeps])
    @assert haskey(timestep_report[:ministeps][ministep], :steps)

    step_reports = timestep_report[:ministeps][ministep][:steps]
    return analyze_convergence(step_reports; kwargs...)

end