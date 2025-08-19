"""
    compute_contraction_factor(r, N)

Compute contraction factor from a number of Newton iterate distances from
convergence (defined with a user-prescribed distance metric). For more than two
iterates, the contraction factor is estimated using least-squares assuming the
iterates follow a geometric series: rₙ = Θⁿr₀. The function also computes the
target contraction factor under this assumption assuming N iterations left.
"""
function compute_contraction_factor(d, N)

    # Use distance from convergence + 1 to avoid division by small numbers
    r = d .+ 1.0
    # Number of iterates used to estimate contraction factor
    n = size(r,1)-1
    # Approximate log10 of contraction factor using least squares
    num = r[1,:].*0.0
    for k = 1:n
        num .+= log.(r[k+1,:]./r[1,:]).^k
    end
    # num = sum(log.(r[2:end,:]./r[1,:]).*(1:n), dims=1)
    den = sum((1:n).^2)
    θ = exp.(num./den)
    # Compute target contraction factor convergence in N iterations
    θ_target = r[1,:].^(-1/N)
    
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
function oscillation(distance)

    d = distance
    (size(d,1) < 3) ? (return falses(size(distance,2))) : nothing

    Δ1 = distance[end-2,:]
    Δ2 = distance[end-1,:]
    Δ3 = distance[end,:]

    osc = Δ3 .> Δ2 .< Δ1 .|| Δ3 .< Δ2 .> Δ1

    return osc

end

function iterations_left(contraction_factor, dist)

    # Compute number of iterations left if we should converge in
    # target_iterations iterations
    θ = contraction_factor
    N = ceil.(-log.(dist)./log.(θ))
    N = max.(N, 0)
    ok = isfinite.(N)
    N[ok] .= Int64.(N[ok])
    return N

end