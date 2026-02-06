"""
    compute_contraction_factor(r, N)

Compute contraction factor from a number of Newton iterate distances from
convergence (defined with a user-prescribed distance metric). For more than two
iterates, the contraction factor is estimated using least-squares assuming the
iterates follow a geometric series: rₙ = Θⁿr₀. The function also computes the
target contraction factor under this assumption assuming N iterations left.
"""
function compute_contraction_factor(distances, N_target)

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
    θ = exp.(num./den)
    # Compute target contraction to converge in N iterations
    θ_target = δ[end-1,:].^(-1/N_target)
    N = -log.(δ[end-1,:])./log.(θ)
    
    return θ, N, θ_target

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

    δ1, δ2, δ3 = distance_1, distace_2, distance_3
    # Check if δ1 > δ2 < δ3 or δ1 < δ2 > δ3
    gt_tol = (x,y) -> (x .- y) .> tol
    lt_tol = (x,y) -> gt_tol(y,x)
    osc_1 = lt_tol(δ2, δ1) .&& lt_tol(δ2, δ3)
    osc_2 = gt_tol(δ2, δ1) .&& gt_tol(δ2, δ3)
    osc = osc_1 .|| osc_2
    # Return oscillation flags
    return osc

end

function oscillation(distances, tol=1e-6)

    # Early return if we only have two iterates
    (size(distances,1) < 3) ? (return falses(size(distances,2))) : nothing
    # Get last three iterates
    δ1, δ2, δ3 = distances[end,:], distances[end-1,:], distances[end-2,:]
    return oscillation(δ1, δ2, δ3, tol)

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