function duo_coefficients(t_i, t_j, l)
    M = @SMatrix [
        t_i[1] t_j[1];
        t_i[2] t_j[2]
    ]
    if abs(det(M)) < 1e-8
        α = β = Inf
    else
        α, β = M\l
    end
    return (α, β)
end

function triplet_coefficients(t_i, t_j, t_k, l)
    M = @SMatrix [
        t_i[1] t_j[1] t_k[1];
        t_i[2] t_j[2] t_k[2];
        t_i[3] t_j[3] t_k[3]
    ]
    if abs(det(M)) < 1e-8
        α = β = γ = Inf
    else
        α, β, γ = M\l
    end
    return (α, β, γ)
end

function find_minimizing_basis_inner(l::SVector{3, Num_t}, all_t, print = false, stop_early = true) where Num_t
    N = length(all_t)
    get_t(i) = all_t[i]
    # Intermediate variables
    best_triplet = (0, 0, 0)
    best_triplet_W = (Inf, Inf, Inf)
    best_value = Inf

    ϵ = 0.0
    for i in 1:(N-2)
        t_i = get_t(i)
        for j in (i+1):(N-1)
            t_j = get_t(j)
            for k in (j+1):N
                t_k = get_t(k)
                α, β, γ = triplet_coefficients(t_i, t_j, t_k, l)
                # @info "$i $j $k" α β γ 
                # @info "Vectors" t_i t_j t_k
                if α ≥ ϵ && β ≥ ϵ && γ ≥ ϵ
                    ijk_value = max(α, β, γ)
                    ijk = (i, j, k)
                    W = (α, β, γ)
                    if ijk_value ≤ 1 && stop_early
                        if print
                            @info "Found optimal triplet." (α, β, γ) ijk t_i t_j t_k l
                        end
                        return (ijk, W)
                    elseif ijk_value < best_value
                        if print
                            @info "Value $ijk_value < $best_value, setting current best value to $ijk"
                        end
                        best_value = ijk_value
                        best_triplet = ijk
                        best_triplet_W = W
                    end
                end
            end
        end
    end
    if print
        @info "Picking best found triplet." best_triplet_W best_triplet
    end
    return (best_triplet, best_triplet_W)
end

function find_minimizing_basis_inner(l::SVector{2, Num_t}, all_t, print = false, stop_early = true) where Num_t
    N = length(all_t)
    get_t(i) = all_t[i]
    # Intermediate variables
    best_triplet = (0, 0, 0)
    best_triplet_W = (Inf, Inf, Inf)
    best_value = Inf

    ϵ = 0.0
    for i in 1:(N-1)
        t_i = get_t(i)
        for j in (i+1):N
            t_j = get_t(j)
            α, β = duo_coefficients(t_i, t_j, l)
            if α ≥ ϵ && β ≥ ϵ
                ijk_value = max(α, β)
                ijk = (i, j)
                W = (α, β)
                if ijk_value ≤ 1 && stop_early
                    if print
                        @info "Found optimal triplet." (α, β) ijk t_i t_j l
                    end
                    return (ijk, W)
                elseif ijk_value < best_value
                    best_value = ijk_value
                    best_triplet = ijk
                    best_triplet_W = W
                    if print
                        @info "Value $ijk_value < $best_value, setting current best value to $ijk"
                    end
                end
            end
        end
    end
    if print
        @info "Picking best found triplet." best_triplet_W best_triplet
    end
    return (best_triplet, best_triplet_W)
end

function candidate_vectors(x_t, x, i; normalize = true)
    t = x[i] - x_t
    if normalize
        t = t./norm(t, 2)
    end
    return t
end

function candidate_vectors(x_t, x; normalize = true)
    t = similar(x)
    for i in eachindex(x)
        t[i] = candidate_vectors(x_t, x, i)
    end
    return t
end

function find_minimizing_basis(x_t::T, l::T, all_x::AbstractVector{T}; check = false, verbose = false, stop_early = true, throw = true) where T
    all_x = copy(all_x)
    all_t = candidate_vectors(x_t, all_x, normalize = true)
    l_norm = norm(l, 2)
    l_bar = l/l_norm
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # function F_sort(i)
    #     return abs(dot(x_t + l_bar, x_t + all_t[i]))
    # end
    p1 = x_t + l_bar
    function F_sort(i)
        v1 = all_t[i]
        v2 = l_bar
        dotnormed = dot(v1, v2)/(norm(v1)*norm(v2))
        # Guard against noise
        dotnormed = clamp(dotnormed, 0.0, 1.0)
        out = acos(dotnormed)
        return out
    end
    sorted_indices = sort(eachindex(all_t), by = F_sort)

    ijk, w = find_minimizing_basis_inner(l_bar, all_t[sorted_indices], verbose, stop_early)
    if any(isequal(0), ijk)
        if throw
            handle_failure(x_t, l, all_x)
        end
    else
        # OK!
        ijk = map(x -> sorted_indices[x], ijk)

        function normalized_weight(i)
            t = candidate_vectors(x_t, all_x, ijk[i], normalize = false)
            return l_norm*w[i]/norm(t, 2)
        end
        w = map(normalized_weight, eachindex(w))
    end
    return (ijk = ijk, w = w)
end

function handle_failure(x_t, l, all_x)
    println("Decomposition failed")
    println("x_t=")
    Base.show(x_t)
    println("\nl=")
    Base.show(l)
    println("\nall_x=")
    Base.show(all_x)
    println("")
    error("Aborting, unable to continue.")
end

function reconstruct_l(indices, weights, x_t, all_x)
    # Reconstruct decomposed l for testing.
    l_r = zero(typeof(x_t))
    for (w, i) in zip(weights, indices)
        t = candidate_vectors(x_t, all_x, i, normalize = false)
        next = w*t
        l_r = l_r .+ next
    end
    return l_r
end
