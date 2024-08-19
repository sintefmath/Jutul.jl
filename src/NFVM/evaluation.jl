function evaluate_flux(p, hf::NFVMLinearDiscretization, ph::Int = 1)
    p_l, p_r = cell_pair_pressures(p, hf, ph)
    T_l = hf.T_left
    T_r = hf.T_right
    q = tpfa_flux(p_l, p_r, hf) + compute_r(p, hf, ph)
    return q
end

function evaluate_flux(p, nfvm::NFVMNonLinearDiscretization, ph::Int = 1)
    L = nfvm.ft_left
    R = nfvm.ft_right

    # We switch sign for the left handed flux to get to the form in the papers.
    r_l = -compute_r(p, L)
    r_r = compute_r(p, R)

    p_l, p_r = cell_pair_pressures(p, nfvm, ph)

    q_l = -tpfa_flux(p_l, p_r, L)
    # Add MPFA part
    q_l += r_l

    q_r = tpfa_flux(p_l, p_r, R)
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
        μ_l = μ_r = 0.5
    else
        μ_l = r_lw/r_total
        μ_r = r_rw/r_total
    end
    q = μ_r*q_l - μ_l*q_r

    return q
end

function compute_r(p::AbstractVector{T}, hf::NFVMLinearDiscretization, ph::Int = 1) where T
    q = zero(T)
    for cell_and_trans in hf.mpfa
        c, T_c = cell_and_trans
        q += p[c]*T_c
    end
    return q
end

function compute_r(p::AbstractMatrix{T}, hf::NFVMLinearDiscretization, ph::Int = 1) where T
    q = zero(T)
    for cell_and_trans in hf.mpfa
        c, T_c = cell_and_trans
        q += p[ph, c]*T_c
    end
    return q
end

function cell_pair_pressures(p::AbstractVector, hf, ph::Int = 1)
    l, r = cell_pair(hf)
    p_l = p[l]
    p_r = p[r]
    return (p_l, p_r)
end

function cell_pair_pressures(p::AbstractMatrix, hf, ph::Int = 1)
    l, r = cell_pair(hf)
    p_l = p[ph, l]
    p_r = p[ph, r]
    return (p_l, p_r)
end

function tpfa_flux(p_l, p_r, hf::NFVMLinearDiscretization)
    T_l = hf.T_left
    T_r = hf.T_right
    return T_l*p_l + T_r*p_r
end