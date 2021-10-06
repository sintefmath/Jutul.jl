# Saturations as primary variable
struct OverallCompositions <: GroupedVariables
    dzMax
    OverallCompositions(dzMax = 0.2) = new(dzMax)
end

degrees_of_freedom_per_entity(model, v::OverallCompositions) =  values_per_entity(model, v) - 1
values_per_entity(model, v::OverallCompositions) = number_of_components(model.system)

maximum_value(::OverallCompositions) = 1.0
minimum_value(::OverallCompositions) = MultiComponentFlash.MINIMUM_COMPOSITION
absolute_increment_limit(z::OverallCompositions) = z.dzMax

function initialize_primary_variable_ad!(state, model, pvar::OverallCompositions, state_symbol, npartials; kwarg...)
    n = values_per_entity(model, pvar)
    v = state[state_symbol]
    state[state_symbol] = unit_sum_init(v, model, npartials, n; kwarg...)
    return state
end

function update_primary_variable!(state, p::OverallCompositions, state_symbol, model, dx)
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


function select_secondary_variables_system!(S, domain, system::CompositionalSystem, formulation)
    nph = number_of_phases(system)
    S[:PhaseMassDensities] = TwoPhaseCompositionalDensities()
    S[:TotalMasses] = TotalMasses()
    S[:FlashResults] = FlashResults(system)
    S[:Saturations] = Saturations()
    S[:Temperature] = ConstantVariables([273.15 + 30.0])
    S[:PhaseViscosities] = ConstantVariables(1e-3*ones(nph)) # 1 cP for all phases by default
end

degrees_of_freedom_per_entity(model, v::MassMobilities) = number_of_phases(model.system)*number_of_components(model.system)


@terv_secondary function update_as_secondary!(flash_results, fr::FlashResults, model, param, Pressure, Temperature, OverallCompositions)
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
        Z = view(OverallCompositions, :, i)
        z = value.(Z)
        # Conditions
        c = (p = p, T = T, z = z)
        # Perform flash
        vapor_frac = flash_2ph!(S, K, eos, c, NaN, method = m, extra_out = false)
        single_phase = isnan(vapor_frac)
        
        if single_phase
            error()
        else
            # Update the actual values
            l, v = f.liquid, f.vapor
            x = l.mole_fractions
            y = v.mole_fractions
            @debug "Info:" Z K v x y
            @. x = liquid_mole_fraction(Z, K, vapor_frac)
            @. y = vapor_mole_fraction(x, K)
            if eltype(x)<:ForwardDiff.Dual
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

@terv_secondary function update_as_secondary!(Sat, s::Saturations, model::SimulationModel{D, S}, param, Pressure, OverallCompositions) where {D, S<:CompositionalSystem}
    # error()
end

@terv_secondary function update_as_secondary!(massmob, m::MassMobilities, model::SimulationModel{D, S}, param) where {D, S<:CompositionalSystem}
    # error()
end

@terv_secondary function update_as_secondary!(rho, m::TwoPhaseCompositionalDensities, model::SimulationModel{D, S}, param, FlashResults) where {D, S<:CompositionalSystem}
    # error()
end


