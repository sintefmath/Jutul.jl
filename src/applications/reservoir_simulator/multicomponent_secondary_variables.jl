# Saturations as primary variable
struct OverallMoleFractions <: FractionVariables
    dz_max
    OverallMoleFractions(;dz_max = 0.2) = new(dz_max)
end

values_per_entity(model, v::OverallMoleFractions) = number_of_components(model.system)

minimum_value(::OverallMoleFractions) = MultiComponentFlash.MINIMUM_COMPOSITION
absolute_increment_limit(z::OverallMoleFractions) = z.dz_max


function update_primary_variable!(state, p::OverallMoleFractions, state_symbol, model, dx)
    s = state[state_symbol]
    unit_sum_update!(s, p, model, dx)
end

struct FlashResults <: ScalarVariable
    storage
    method
    function FlashResults(system; method = SSIFlash(), kwarg...)
        eos = system.equation_of_state
        # np = number_of_partials_per_entity(system, Cells())
        n = MultiComponentFlash.number_of_components(eos)
        s = flash_storage(eos, method = method, inc_jac = true, diff_externals = true, npartials = n)
        new(s, method)
    end
end

default_value(model, ::FlashResults) = FlashedMixture2Phase(model.system.equation_of_state)

function initialize_variable_value!(state, model, pvar::FlashResults, symb, val::AbstractDict; need_value = false)
    @assert need_value == false
    v = default_value(model, pvar)
    V = repeat([v], number_of_entities(model, pvar))
    initialize_variable_value!(state, model, pvar, symb, V)
end

function initialize_variable_ad(state, model, pvar::FlashResults, symb, npartials, diag_pos; context = DefaultContext(), kwarg...)
    n = number_of_entities(model, pvar)
    v_ad = get_ad_entity_scalar(1.0, npartials, diag_pos; kwarg...)
    ∂T = typeof(v_ad)

    r = FlashedMixture2Phase(model.system.equation_of_state, ∂T)
    state[symb] = repeat([r], n)
    return state
end

struct TwoPhaseCompositionalDensities <: PhaseMassDensities
end

struct PhaseMassFractions <: FractionVariables
    phase
end

values_per_entity(model, v::PhaseMassFractions) = number_of_components(model.system)

function select_secondary_variables_system!(S, domain, system::CompositionalSystem, formulation)
    nph = number_of_phases(system)
    S[:PhaseMassDensities] = TwoPhaseCompositionalDensities()
    S[:LiquidMassFractions] = PhaseMassFractions(:liquid)
    S[:VaporMassFractions] = PhaseMassFractions(:vapor)
    S[:TotalMasses] = TotalMasses()
    S[:FlashResults] = FlashResults(system)
    S[:Saturations] = Saturations()
    S[:Temperature] = ConstantVariables([273.15 + 30.0])
    S[:PhaseViscosities] = ConstantVariables(1e-3*ones(nph)) # 1 cP for all phases by default
end

degrees_of_freedom_per_entity(model, v::MassMobilities) = number_of_phases(model.system)#*number_of_components(model.system)


@terv_secondary function update_as_secondary!(flash_results, fr::FlashResults, model, param, Pressure, Temperature, OverallMoleFractions)
    # S = flash_storage(eos)
    S = fr.storage
    m = fr.method
    eos = model.system.equation_of_state
    @time for (i, f) in enumerate(flash_results)
        K = f.K
        P = Pressure[i]
        T = Temperature[1, i]
        # Grab values
        p = value(P)
        t = value(T)
        Z = view(OverallMoleFractions, :, i)
        z = value.(Z)
        # Conditions
        c = (p = p, T = T, z = z)
        # Perform flash
        vapor_frac = flash_2ph!(S, K, eos, c, NaN, method = m, extra_out = false)

        l, v = f.liquid, f.vapor
        x = l.mole_fractions
        y = v.mole_fractions

        if isnan(vapor_frac)
            # Single phase condition. Life is easy.
            Z_L = mixture_compressibility_factor(eos, (p = P, T = T, z = Z))
            Z_V = Z_L
            @. x = Z
            @. y = Z
            V = single_phase_label(eos.mixture, c)
            if V > 0.5
                phase_state = SinglePhaseVapor()
            else
                phase_state = SinglePhaseLiquid()
            end
        else
            # Two-phase condition: We have some work to do.
            @. x = liquid_mole_fraction(Z, K, vapor_frac)
            @. y = vapor_mole_fraction(x, K)
            if eltype(x)<:ForwardDiff.Dual
                inverse_flash_update!(S, eos, c, vapor_frac)
                ∂c = (p = P, T = T, z = Z)
                V = set_partials_vapor_fraction(convert(eltype(x), vapor_frac), S, eos, ∂c)
                set_partials_phase_mole_fractions!(x, S, eos, ∂c, :liquid)
                set_partials_phase_mole_fractions!(y, S, eos, ∂c, :vapor)
            else
                V = vapor_frac
            end
            # TODO: Fix so these don't allocate memory
            Z_L = mixture_compressibility_factor(eos, (p = P, T = T, z = x))
            Z_V = mixture_compressibility_factor(eos, (p = P, T = T, z = y))

            phase_state = TwoPhaseLiquidVapor()
        end
        flash_results[i] = FlashedMixture2Phase(phase_state, K, V, x, y, Z_L, Z_V)
    end
end

@terv_secondary function update_as_secondary!(Sat, s::Saturations, model::SimulationModel{D, S}, param, FlashResults) where {D, S<:CompositionalSystem}
    for i in 1:size(Sat, 2)
        S_l, S_v = phase_saturations(FlashResults[i])
        Sat[1, i] = S_l
        Sat[2, i] = S_v
    end
end

@terv_secondary function update_as_secondary!(massmob, m::MassMobilities, model::SimulationModel{D, S}, param) where {D, S<:CompositionalSystem}
    # error()
end

@terv_secondary function update_as_secondary!(X, m::PhaseMassFractions, model::SimulationModel{D, S}, param, FlashResults) where {D, S<:CompositionalSystem}
    molar_mass = map((x) -> x.mw, model.system.equation_of_state.mixture.properties)
    phase = m.phase
    for (i, f) in enumerate(FlashResults)
        if phase_is_present(phase, f.state)
            X_i = view(X, :, i)
            r = getfield(f, phase)
            x_i = r.mole_fractions
            update_mass_fractions!(X_i, x_i, molar_mass)
        end
    end
end

function update_mass_fractions!(X, x, molar_masses)
    t = 0
    for i in 1:length(x)
        tmp = molar_masses[i]*x[i]
        t += tmp
        X[i] = tmp
    end
    @. X = X/t
end

@terv_secondary function update_as_secondary!(rho, m::TwoPhaseCompositionalDensities, model::SimulationModel{D, S}, param, Pressure, Temperature, OverallMoleFractions, FlashResults) where {D, S<:CompositionalSystem}
    eos = model.system.equation_of_state
    for i in 1:size(rho, 2)
        p = Pressure[i]
        T = Temperature[1, i]
        z = view(OverallMoleFractions, :, i)
        cond = (p = p, T = T, z = z)

        ρ_l, ρ_v = mass_densities(eos, cond, FlashResults[i])
        rho[1, i] = ρ_l
        rho[2, i] = ρ_v
    end
end

# Total masses
@terv_secondary function update_as_secondary!(totmass, tv::TotalMasses, model::SimulationModel{G, S}, param, FlashResults, PhaseMassDensities, Saturations, VaporMassFractions, LiquidMassFractions) where {G, S<:CompositionalSystem}
    pv = get_pore_volume(model)
    ρ = PhaseMassDensities
    X = LiquidMassFractions
    Y = VaporMassFractions
    Sat = Saturations
    F = FlashResults

    # @info "Total mass" value.(Sat) value.(X) value.(Y) value.(ρ)
    @tullio totmass[c, i] = two_phase_compositional_mass(F[i].state, ρ, X, Y, Sat, c, i)*pv[i]
    # @debug "Total mass updated:" totmass'
end

function two_phase_compositional_mass(::SinglePhaseVapor, ρ, X, Y, Sat, c, i)
    M = ρ[2, i]*Sat[2, i]*Y[c, i]
    # @info "Vapor cell $i component $c" M
    return M
end

function two_phase_compositional_mass(::SinglePhaseLiquid, ρ, X, Y, Sat, c, i)
    M = ρ[1, i]*Sat[1, i]*X[c, i]
    # @info "Liquid cell $i component $c" M X[c, i] Sat[1, i] ρ[:, i]
    return M
end

function two_phase_compositional_mass(::TwoPhaseLiquidVapor, ρ, X, Y, S, c, i)
    M_v = ρ[1, i]*S[1, i]*X[c, i]
    M_l = ρ[2, i]*S[2, i]*Y[c, i]
    # @info "Two-phase cell $i component $c" M_v M_l
    return M_l + M_v
end