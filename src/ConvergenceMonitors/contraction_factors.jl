"""
    θ, N, θ_target = compute_contraction_factor(distances, N_target)

Estimate the contraction factor from a number of consecutive Newton iterate
distances from convergence (rows of `distances`, one column per residual
measure). The iterates are assumed to follow a geometric series, δₖ = θᵏ⁻¹δ₁,
and θ is estimated by least-squares regression on the log-transformed distances
(for two iterates, this reduces to θ = δ₂/δ₁).

Returns the estimated contraction factor `θ`, the estimated number of iterations
`N` needed to reach convergence (δ = 1) from the second-to-last iterate given
`θ`, and the target contraction factor `θ_target` required to converge from the
second-to-last iterate in `N_target` iterations.
"""
function compute_contraction_factor(distances, N_target)

    # Number of iterates used to estimate contraction factor
    δ = distances
    n = size(δ, 1)
    # Use least-squares regression on log-transformed contraction factors:
    # minₖ ∑(θ^{k-1} - rₖ/r₁)² ~ minₖ ∑[(k-1)log(θ) - log(rₖ/r₁)]²
    # ⇒ log(θ) ∑(k-1)² = ∑(k-1)log(rₖ/r₁)
    num, den = zeros(size(δ, 2)), 0.0
    for k in 1:n
        @views num .+= (k - 1) .* log.(δ[k, :] ./ δ[1, :])
        den += (k - 1)^2
    end
    θ = exp.(num ./ den)
    # Compute target contraction to converge in N iterations
    δ_prev = @view δ[end - 1, :]
    θ_target = δ_prev .^ (-1 / N_target)
    N = -log.(δ_prev) ./ log.(θ)

    return θ, N, θ_target

end

"""
    oscillation(distance_1, distance_2, distance_3, tol = 1.0e-6)

Check if three consecutive distance vectors are oscillating. Returns `true` for
entries where the middle iterate `distance_2` is a strict local minimum or
maximum, with differences exceeding `tol`.
"""
function oscillation(distance_1, distance_2, distance_3, tol = 1.0e-6)

    δ1, δ2, δ3 = distance_1, distance_2, distance_3
    # Check if δ1 > δ2 < δ3 or δ1 < δ2 > δ3
    gt_tol = (x, y) -> (x .- y) .> tol
    lt_tol = (x, y) -> gt_tol(y, x)
    osc_1 = lt_tol(δ2, δ1) .&& lt_tol(δ2, δ3)
    osc_2 = gt_tol(δ2, δ1) .&& gt_tol(δ2, δ3)
    osc = osc_1 .|| osc_2
    # Return oscillation flags
    return osc

end

"""
    oscillation(distances, tol = 1.0e-6)

Check if the recent distance history is oscillating, using the last three rows
of `distances` (see the three-argument method). If fewer than three iterates are
provided, the function returns `false` for all entries.
"""
function oscillation(distances, tol = 1.0e-6)

    # Early return if we only have two iterates
    if size(distances, 1) < 3
        return falses(size(distances, 2))
    end
    # Get last three iterates
    @views δ1, δ2, δ3 = distances[end, :], distances[end - 1, :], distances[end - 2, :]
    return oscillation(δ1, δ2, δ3, tol)

end
