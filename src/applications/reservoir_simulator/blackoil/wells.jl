function apply_well_reservoir_sources!(sys::BlackOilSystem, res_q, well_q, state_res, state_well, param_res, param_well, perforations, sgn)
    p_res = state_res.Pressure
    p_well = state_well.Pressure

    val = x -> local_ad(x, nothing)

    μ = state_res.PhaseViscosities
    kr = state_res.RelativePermeabilities
    ρ = state_res.PhaseMassDensities
    rs = state_res.Rs
    b = state_res.ShrinkageFactors

    ρ_w = state_well.PhaseMassDensities
    s_w = state_well.Saturations
    rs_w = state_well.Rs
    b_w = state_well.ShrinkageFactors

    rhoS = tuple(param_res[:reference_densities]...)

    perforation_sources_blackoil!(well_q, perforations, val(p_res),         p_well,  val(kr), val(μ), val(ρ), val(b), val(rs),    ρ_w,       b_w,     s_w, rs_w, sgn, rhoS)
    perforation_sources_blackoil!(res_q,  perforations,     p_res,      val(p_well),     kr,      μ,      ρ,      b,      rs, val(ρ_w), val(b_w), val(s_w), val(rs_w), sgn, rhoS)
end

function perforation_sources_blackoil!(target, perf, p_res, p_well, kr, μ, ρ, b, rs, ρ_w, b_w, s_w, rs_w, sgn, rhoS)
    # (self -> local cells, reservoir -> reservoir cells, WI -> connection factor)
    nc = size(ρ, 1)
    nph = size(μ, 1)
    a, l, v = 1, 2, 3

    rhoOS = rhoS[a]
    rhoGS = rhoS[v]
    @inbounds for i in eachindex(perf.self)
        si, ri, wi, gdz = unpack_perf(perf, i)
        if gdz != 0
            ρ_mix = @views mix_by_saturations(s_w[:, si], ρ_w[:, si])
            ρgdz = gdz*ρ_mix
        else
            ρgdz = 0
        end
        @inbounds dp = wi*(p_well[si] - p_res[ri] + ρgdz)
        if dp > 0
            # Injection
            λ_t = 0
            @inbounds for ph in 1:nph
                λ_t += kr[ph, ri]/μ[ph, ri]
            end
            Q = sgn*λ_t*dp

            bO = b_w[l, si]
            bG = b_w[v, si]
            rs_i = rs_w[si]

            target[a, i] = s_w[a, si]*ρ_w[a, si]*Q
            target[l, i] = rhoOS*bO*Q
            target[v, i] = rhoGS*(bG + rs_i*bO)*Q
        else
            # Production
            Q = sgn*dp
            target[a, i] = Q*ρ[a, ri]*kr[a, ri]/μ[a, ri]

            α_l = b[l, ri]*kr[l, ri]/μ[l, ri]
            α_v = b[v, ri]*kr[v, ri]/μ[v, ri]

            target[l, i] = α_l*rhoOS
            target[v, i] = (α_l*rs[ri] + α_v)*rhoGS
        end
    end
end


function flash_wellstream_at_surface(well_model::SimulationModel{D, S}, well_state, rhoS) where {D, S<:BlackOilSystem}
    vol = well_state.TotalMasses[:, 1]./rhoS
    volfrac = vol./sum(vol)
    return (rhoS, volfrac)
end
