Base.@kwdef struct GasMassFraction <: ScalarVariable
    dz_max = 0.1
end

maximum_value(::GasMassFraction) = 1.0
minimum_value(::GasMassFraction) = 0.0
absolute_increment_limit(s::GasMassFraction) = s.dz_max

struct BlackOilPhaseState <: ScalarVariable

end

default_value(model, ::BlackOilPhaseState) = OilAndGas
initialize_secondary_variable_ad!(state, model, var::BlackOilPhaseState, arg...; kwarg...) = state

max_dissolved_gas_fraction(rs, rhoOS, rhoGS) = rs*rhoGS/(rhoOS + rs*rhoGS)

@terv_secondary function update_as_secondary!(phase_state, m::BlackOilPhaseState, model::SimulationModel{D, S}, param, Pressure, GasMassFraction) where {D, S<:BlackOilSystem}
    tab = model.system.saturation_table
    rhoS = param[:reference_densities]
    rhoOS = rhoS[2]
    rhoGS = rhoS[3]
    for i in eachindex(phase_state)
        p = Pressure[i]
        z_g = GasMassFraction[i]
        z_g_bub = max_dissolved_gas_fraction(tab(p), rhoOS, rhoGS)
        if z_g_bub < z_g
            phase_state[i] = OilAndGas
        else
            phase_state[i] = OilOnly
        end
    end
end

struct Rs <: ScalarVariable end

@terv_secondary function update_as_secondary!(rs, m::Rs, model::SimulationModel{D, S}, param, PhaseState, Pressure, GasMassFraction) where {D, S<:BlackOilSystem}
    tab = model.system.saturation_table
    rhoS = param[:reference_densities]
    rhoOS = rhoS[2]
    rhoGS = rhoS[3]
    for i in eachindex(PhaseState)
        p = Pressure[i]
        if PhaseState[i] == OilAndGas
            z_g = max_dissolved_gas_fraction(tab(p), rhoOS, rhoGS)
        else
            z_g = GasMassFraction[i]
        end
        rs[i] = rhoOS*z_g/(rhoGS*(1-z_g))
    end
end

@terv_secondary function update_as_secondary!(b, ρ::DeckShrinkageFactors, model::SimulationModel{D, StandardBlackOilSystem{T, true}}, param, Pressure, Rs) where {D, T}
    pvt, reg = ρ.pvt, ρ.regions
    # Note immiscible assumption
    tb = thread_batch(model.context)
    nph, nc = size(b)

    w = 1
    g = 3
    o = 2
    bO = pvt[o]
    bG = pvt[g]
    bW = pvt[w]
    @batch minbatch = tb for i in 1:nc
        p = Pressure[i]
        rs = Rs[i]
        b[w, i] = shrinkage(bW, reg, p, i)
        b[o, i] = shrinkage(bO, reg, p, rs, i)
        b[g, i] = shrinkage(bG, reg, p, i)
    end
end

@terv_secondary function update_as_secondary!(b, ρ::DeckViscosity, model::SimulationModel{D, StandardBlackOilSystem{T, true}}, param, Pressure, Rs) where {D, T}
    pvt, reg = ρ.pvt, ρ.regions
    # Note immiscible assumption
    tb = thread_batch(model.context)
    nph, nc = size(b)

    w = 1
    g = 3
    o = 2
    bO = pvt[o]
    bG = pvt[g]
    bW = pvt[w]
    @batch minbatch = tb for i in 1:nc
        p = Pressure[i]
        rs = Rs[i]
        b[w, i] = viscosity(bW, reg, p, i)
        b[o, i] = viscosity(bO, reg, p, rs, i)
        b[g, i] = viscosity(bG, reg, p, i)
    end
end

# @terv_secondary function update_as_secondary!(rho, m::DeckDensity, model::SimulationModel{D, S}, param, Pressure, Rs) where {D, S<:BlackOilSystem}
#     sys = model.system
#     rhos = param[:reference_densities]
#     pvt, reg = ρ.pvt, ρ.regions

#     # eos = sys.equation_of_state
#     n = size(rho, 2)
#     for i = 1:n
#         rs = Rs[i]
#         p = Pressure[i]
#         rho[1, i] = oil_density(opvt, p, rs)
#         rho[2, i] = gas_density(gpvt, p)
#     end
# end

# @terv_secondary function update_as_secondary!(mu, μ::DeckViscosity, model::SimulationModel{D, S}, param, Pressure, Rs) where {D, S<:BlackOilSystem}
#     pvt, reg = μ.pvt, μ.regions
#     tb = thread_batch(model.context)
#     nph, nc = size(mu)
#     for ph in 1:nph
#         pvt_ph = pvt[ph]
#         @batch minbatch = tb for i in 1:nc
#             p = Pressure[i]
#             @inbounds mu[ph, i] = viscosity(pvt_ph, reg, p, i)
#         end
#     end
# end
