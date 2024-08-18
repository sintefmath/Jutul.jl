function evaluate_flux(p, hf::NFVMLinearDiscretization)
    p_l = p[hf.left]
    p_r = p[hf.right]
    T_l = hf.T_left
    T_r = hf.T_right
    q = T_l*p_l + T_r*p_r
    for cell_and_trans in hf.mpfa
        c, T = cell_and_trans
        q += p[c]*T
    end
    return q
end

function evaluate_flux(p::AbstractVector{T}, nfvm::NFVMNonLinearDiscretization) where {T}
    L = nfvm.ft_left
    R = nfvm.ft_right
    l, r = cell_pair(L)

    # We switch sign for the left handed flux to get to the form in the papers.
    r_l = -compute_r(p, L)
    r_r = compute_r(p, R)

    p_l = p[l]
    p_r = p[r]

    T_ll = -L.T_left
    T_lr = -L.T_right
    q_l = T_ll*p_l + T_lr*p_r
    # Add MPFA part
    q_l += r_l

    T_rl = R.T_left
    T_rr = R.T_right
    q_r = T_rl*p_l + T_rr*p_r
    # Add MPFA part
    q_r += r_r

    if nfvm.scheme == :ntpfa
        r_lw = r_l
        r_rw = r_r
    else
        # TODO: Double check this.
        r_lw = abs(r_l)
        r_rw = abs(r_r)
    end
    r_total = r_lw + r_rw
    if abs(r_total) < 1e-10
        μ_l = μ_r = convert(T, 0.5)
    else
        μ_l = r_lw/r_total
        μ_r = r_rw/r_total
    end
    q = μ_r*q_l - μ_l*q_r

    return q
end

function compute_r(p::AbstractVector{T}, hf::NFVMLinearDiscretization) where {T}
    q = zero(T)
    for cell_and_trans in hf.mpfa
        c, T_c = cell_and_trans
        q += p[c]*T_c
    end
    return q
end
