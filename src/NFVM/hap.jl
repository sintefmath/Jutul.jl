function find_harmonic_average_point(K_1, x_1, K_2, x_2, x_f, n_f)
    @assert length(x_1) == length(x_2) == length(n_f) == length(x_f)

    λ_1, γ_1 = compute_coefficients(K_1, n_f)
    y_1, d_1 = project_point_to_plane(x_1, x_f, n_f)

    λ_2, γ_2 = compute_coefficients(K_2, n_f)
    y_2, d_2 = project_point_to_plane(x_2, x_f, n_f)

    w_1 = λ_1*d_2
    w_2 = λ_2*d_1
    w_t = w_1 + w_2
    avg_pt = (w_1*y_1 + w_2*y_2 + d_1*d_2*(γ_1 - γ_2))/w_t
    return (avg_pt, (w_1/w_t, w_2/w_t))
end

function compute_coefficients(K, n)
    λ = n'*K*n
    γ = K*n - λ*n
    return (λ, γ)
end

function project_point_to_plane(pt, pt_on_plane, plane_normal)
    # point_on_plane = pt - cross(dot(normal, pt - pt_plane), normal)
    # dist_to_plane = norm(pt - point_on_plane, 2)
    dist_to_plane = dot(pt_on_plane - pt, plane_normal)
    pt_projected_onto_plane = pt + dist_to_plane * plane_normal
    return (pt_projected_onto_plane, abs(dist_to_plane))
end
