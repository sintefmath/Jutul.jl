"""
    compute_distance(report; distance_function = r -> scaled_residual_norm(r), mapping = v -> maximum(v))

Compute distance from convergence using a user-defined distance function, and
optionally apply a mapping to the distance. The function returns the distance
and the names of equation residual norms used in the distance computation.
"""
function compute_distance(report; 
    distance_function = r -> scaled_residual_norm(r),
    mapping = v -> [maximum(v)]
    )
    
    # Compute distance using distance function
    distance, names = distance_function(report)
    # Apply transformation
    distance = mapping(distance)

    return distance, names

end

"""
    scaled_residual_norm(report)

Compute distance to convergence as the residual norm of the equations scaled by
their respective tolerances.
"""
function scaled_residual_norm(report)

    residuals = get_multimodel_residuals(report)
    values, names = flatten_dict(residuals)
    distance = max.(values .- 1.0, 0.0)

    return distance, names

end

"""
    nonconverged_equations(report)

Compute distance to convergence as non-converged equations, (1.0 for
non-converged and 0.0 for converged). The final distance is typically taken as
the sum of the output from this function.
"""
function nonconverged_equations(report)

    values, names = scaled_residual_norm(report)
    distance = (values .> 0.0) .+ 0.0

    return distance, names

end