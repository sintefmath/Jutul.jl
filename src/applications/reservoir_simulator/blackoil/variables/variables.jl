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
        p0 = phase_state[i]
        p = Pressure[i]
        z_g = GasMassFraction[i]
        z_g_bub = max_dissolved_gas_fraction(tab(p), rhoOS, rhoGS)
        if z_g_bub < z_g
            new_state = OilAndGas
        else
            new_state = OilOnly
        end
        phase_state[i] = new_state
        if p0 != new_state
            # @info "Switching cell $i from $p0 to $new_state"
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

@terv_secondary function update_as_secondary!(rho, m::DeckDensity, model::SimulationModel{D, S}, param, Rs, ShrinkageFactors) where {D, S<:BlackOilSystem}
    # sys = model.system
    b = ShrinkageFactors
    rhoS = param[:reference_densities]
    rhoWS = rhoS[1]
    rhoOS = rhoS[2]
    rhoGS = rhoS[3]
    # pvt, reg = ρ.pvt, ρ.regions
    # eos = sys.equation_of_state
    w = 1
    g = 3
    o = 2
    n = size(rho, 2)
    for i = 1:n
        rho[w, i] = b[w, i]*rhoWS
        rho[o, i] = b[o, i]*(rhoOS + Rs[i]*rhoGS)
        rho[g, i] = b[g, i]*rhoGS
    end
end

@terv_secondary function update_as_secondary!(totmass, tv::TotalMasses, model::SimulationModel{G,S}, param,
                                                                                                    Rs,
                                                                                                    ShrinkageFactors,
                                                                                                    PhaseMassDensities,
                                                                                                    Saturations,
                                                                                                    FluidVolume) where {G,S<:BlackOilSystem}
    rhoS = tuple(param[:reference_densities]...)
    tb = thread_batch(model.context)
    sys = model.system
    nc = size(totmass, 2)
    # @batch minbatch = tb for cell = 1:nc
    for cell = 1:nc
        @inbounds @views blackoil_mass!(totmass[:, cell], FluidVolume, PhaseMassDensities, Rs, ShrinkageFactors, Saturations, rhoS, cell, (1,2,3))
    end
end

function blackoil_mass!(M, pv, ρ, Rs, b, S, rhoS, cell, phase_indices)
    a, l, v = phase_indices
    bO = b[l, cell]
    bG = b[v, cell]
    rs = Rs[cell]
    sO = S[l, cell]
    sG = S[v, cell]
    Φ = pv[cell]

    # Water is trivial
    M[a] = Φ*ρ[a, cell]*S[a, cell]
    # Oil is only in oil phase
    M[l] = Φ*rhoS[l]*bO*sO
    # Gas is in both phases
    M[v] = Φ*rhoS[v]*(bG*sG + sO*bO*rs)
end

@terv_secondary function update_as_secondary!(s, SAT::Saturations, model::SimulationModel{D, S}, param, ImmiscibleSaturation, PhaseState, GasMassFraction, ShrinkageFactors, Rs) where {D, S<:BlackOilSystem}
    # tb = thread_batch(model.context)
    nph, nc = size(s)
    a, l, v = 1, 2, 3
    rhoS = param[:reference_densities]
    rhoOS = rhoS[l]
    rhoGS = rhoS[v]
    for i = 1:nc
        sw = ImmiscibleSaturation[i]
        s[a, i] = sw
        if PhaseState[i] == OilAndGas
            rs = Rs[i]
            bO = ShrinkageFactors[l, i]
            bG = ShrinkageFactors[v, i]

            zo = 1 - GasMassFraction[i]
            so = zo*rhoGS*bG/(rhoOS*bO + zo*(rhoGS*bG - rhoOS*bO - rhoGS*bO*rs))
        else
            so = 1
        end
        s[a, i] = sw
        s[l, i] = (1 - sw)*so
        s[v, i] = (1 - sw)*(1 - so)
    end
end
