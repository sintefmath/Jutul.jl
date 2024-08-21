function evaluate_flux(p, hf::NFVMLinearDiscretization{T}, ph::Int) where {T}
    p_l, p_r = cell_pair_pressures(p, hf, ph)
    T_l = hf.T_left
    T_r = hf.T_right
    q = tpfa_flux(p_l, p_r, hf) + compute_r(p, hf, ph)
    return q
end

function evaluate_flux(p, nfvm, ph::Int)
    L = nfvm.ft_left
    R = nfvm.ft_right

    p_l, p_r = cell_pair_pressures(p, nfvm, ph)
    # Fluxes from different side with different sign
    q_l, r_l = ntpfa_half_flux(p_l, p_r, p, L, ph)
    q_r, r_r = ntpfa_half_flux(p_l, p_r, p, R, ph, -1)

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
        μ_l = r_rw/r_total
        μ_r = r_lw/r_total
    end

    q = μ_l*q_l - μ_r*q_r
    return q
end

function ntpfa_half_flux(p_l, p_r, p, disc, ph, sgn = 1)
    q = tpfa_flux(p_l, p_r, disc)
    # Add MPFA part
    r = compute_r(p, disc, ph)
    q += r
    return (sgn*q, sgn*r)
end

function compute_r(p::AbstractVector{T}, hf::NFVMLinearDiscretization, ph::Int) where T
    q = zero(T)
    for cell_and_trans in hf.mpfa
        c, T_c = cell_and_trans
        q += p[c]*T_c
    end
    return q
end

function compute_r(p::AbstractMatrix{T}, hf::NFVMLinearDiscretization, ph::Int) where T
    q = zero(T)
    for cell_and_trans in hf.mpfa
        c, T_c = cell_and_trans
        q += p[ph, c]*T_c
    end
    return q
end

function cell_pair_pressures(p::AbstractVector, hf, ph::Int)
    l, r = cell_pair(hf)
    p_l = p[l]
    p_r = p[r]
    return (p_l, p_r)
end

function cell_pair_pressures(p::AbstractMatrix, hf, ph::Int)
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