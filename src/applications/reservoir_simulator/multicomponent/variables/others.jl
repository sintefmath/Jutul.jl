struct PhaseMassFractions{T} <: CompositionalFractions
    phase::T
end


@terv_secondary function update_as_secondary!(X, m::PhaseMassFractions, model::SimulationModel{D,S}, param, FlashResults) where {D,S<:CompositionalSystem}
    molar_mass = map((x) -> x.mw, model.system.equation_of_state.mixture.properties)
    phase = m.phase
    tb = thread_batch(model.context)
    @inbounds @batch minbatch = tb for i in eachindex(FlashResults)
        f = FlashResults[i]
        if phase_is_present(phase, f.state)
            X_i = view(X, :, i)
            r = phase_data(f, phase)
            x_i = r.mole_fractions
            update_mass_fractions!(X_i, x_i, molar_mass)
        end
    end
end

@inline function update_mass_fractions!(X, x, molar_masses)
    t = zero(eltype(X))
    @inbounds for i in eachindex(x)
        tmp = molar_masses[i] * x[i]
        t += tmp
        X[i] = tmp
    end
    @. X = X / t
end

# Total masses
@terv_secondary function update_as_secondary!(totmass, tv::TotalMasses, model::SimulationModel{G,S}, param,
                                                                                                    FlashResults,
                                                                                                    PhaseMassDensities,
                                                                                                    Saturations,
                                                                                                    VaporMassFractions,
                                                                                                    LiquidMassFractions,
                                                                                                    FluidVolume) where {G,S<:CompositionalSystem}
                                                                                                    pv = FluidVolume
                                                                                                    ρ = PhaseMassDensities
                                                                                                    X = LiquidMassFractions
                                                                                                    Y = VaporMassFractions
                                                                                                    Sat = Saturations
                                                                                                    F = FlashResults

    @tullio totmass[c, i] = two_phase_compositional_mass(F[i].state, ρ, X, Y, Sat, c, i) * pv[i]
end

function degrees_of_freedom_per_entity(model::SimulationModel{G,S}, v::TotalMasses) where {G<:Any,S<:MultiComponentSystem}
    number_of_components(model.system)
end

function two_phase_compositional_mass(state, ρ, X, Y, S, c, i)
    T = eltype(ρ)
    if liquid_phase_present(state)
        @inbounds M_l = ρ[1, i] * S[1, i] * X[c, i]
    else
        M_l = zero(T)
    end

    if vapor_phase_present(state)
        @inbounds M_v = ρ[2, i] * S[2, i] * Y[c, i]
    else
        M_v = zero(T)
    end
    return M_l + M_v
end

