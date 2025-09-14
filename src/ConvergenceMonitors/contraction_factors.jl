"""
    compute_contraction_factor(r, N)

Compute contraction factor from a number of Newton iterate distances from
convergence (defined with a user-prescribed distance metric). For more than two
iterates, the contraction factor is estimated using least-squares assuming the
iterates follow a geometric series: rₙ = Θⁿr₀. The function also computes the
target contraction factor under this assumption assuming N iterations left.
"""
function compute_contraction_factor(distances, N)

    # Number of iterates used to estimate contraction factor
    δ = distances
    n = size(δ,1)
    # Use least-squares regression on log-transformed contraction factors:
    # minₖ ∑(θ^{k-1} - rₖ/r₁)² ~ minₖ ∑[(k-1)log(θ) - log(rₖ/r₁)]²
    # ⇒ log(θ) ∑(k-1)² = ∑(k-1)log(rₖ/r₁)
    num, den = zeros(size(δ,2)), 0.0
    for k = 1:n
        num .+= (k-1).*log.(δ[k,:]./δ[1,:])
        den += (k-1)^2
    end
    θ = exp.(num./max(den,1))
    # Compute target contraction to converge in N iterations
    θ_target = δ[end,:].^(-1/N)
    
    return θ, θ_target

end

"""
    oscillation(contraction_factors, tol)

Check if the contraction factors are oscillating. The function checks if the
contraction factors are oscillating by checking if the last three contraction
factors are below a user-defined tolerance defining "slow" convergece. The
function returns true if the contraction factors are oscillating, and false
otherwise. If less than three contraction factors are provided, the function
returns false.
"""
function oscillation(distance_1, distace_2, distance_3, tol=1e-6)

    # # Early return if we only have two iterates
    # (size(distance,1) < 3) ? (return falses(size(distance,2))) : nothing
    # Get last three distances to convergence
    δ_1, δ_2, δ_3 = distance_1, distace_2, distance_3
    # Check if Δ1 > Δ2 < Δ3 or Δ1 < Δ2 > Δ3
    gt_tol = (x,y) -> (x .- y) .> tol
    lt_tol = (x,y) -> gt_tol(y,x)
    osc_1 = lt_tol(δ_2, δ_1) .&& lt_tol(δ_2, δ_3)
    osc_2 = gt_tol(δ_2, δ_1) .&& gt_tol(δ_2, δ_3)
    osc = osc_1 .|| osc_2

    return osc

end

function iterations_left(contraction_factor, dist)

    # Compute number of iterations left if we should converge in
    # target_iterations iterations
    θ = contraction_factor
    r = dist .+ 1
    N = ceil.(-log.(r)./log.(θ))
    N = max.(N, 0)
    return N

end

function classify_iterate(theta, theta_target, theta_slow, theta_fast, distance, oscillation=missing)

    # Shorthand notation
    θ, θ_t, θ_s, θ_f = theta, theta_target, theta_slow, theta_fast
    osc = ismissing(oscillation) ? false : oscillation
    # Check convergence for all distances
    conv = distance .<= 1.0
    # Check contraction magnitude
    fast = (θ .<= max.(θ_t, θ_f))
    ok = max.(θ_t, θ_f) .< θ .<= θ_s
    div = θ .> θ_s
    # Exclude "diverging" distances that have converged (e.g., θ = 1/1)
    div = div .&& .!conv
    # Classify fast contractions as ok if oscillating
    ok = ok .|| (fast .&& osc)
    # Classify converged distances as fast and require no oscillation
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

function update_violation_count!(num_violations, status; count_type=:per_measure)

    if isnothing(num_violations)
        if count_type == :per_measure
            num_violations = zeros(length(status))
        else
            num_violations = [0]
        end
    end

    fast = status .== 1
    diverge = status .== -1
    if count_type == :per_measure
        num_violations[fast] .-= 1
        num_violations[diverge] .+= 1
    elseif count_type == :total
        num_violations .+= sum(diverge) - sum(fast)
    elseif count_type == :overall
        if all(fast)
            num_violations .-= 1
        elseif any(diverge)
            num_violations .+=1
        end
    end

    num_violations = max.(num_violations, 0)

    return num_violations

end