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
    function FlashResults(domain, system; method = SSIFlash())
        eos = model.equation_of_state
        np = number_of_partials_per_entity(model, Cells())
        s = flash_storage(eos, method = method, inc_jac = true, diff_externals = true, npartials = np)
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
    S[:FlashResults] = FlashResults()
    S[:Saturations] = Saturations()
    S[:Temperature] = ConstantVariables([273.15 + 30.0])
    S[:PhaseViscosities] = ConstantVariables(1e-3*ones(nph)) # 1 cP for all phases by default
end

degrees_of_freedom_per_entity(model, v::MassMobilities) = number_of_phases(model.system)*number_of_components(model.system)


@terv_secondary function update_as_secondary!(f, fr::FlashResults, model, param, Pressure, Temperature, OverallCompositions)
    # S = flash_storage(eos)
    for i in eachindex(f)
        p = value(Pressure[i])
        T = value(Temperature[1, i]) #Constant var hack
        z = value.(OverallCompositions[:, i])
        V, K, = flash_2ph!(S, K, eos, c, NaN, method = m, extra_out = true)
    end
    x + y + z
    error() 
end

@terv_secondary function update_as_secondary!(Sat, s::Saturations, model::SimulationModel{D, S}, param, Pressure, OverallCompositions) where {D, S<:CompositionalSystem}
    error()
end

@terv_secondary function update_as_secondary!(massmob, m::MassMobilities, model::SimulationModel{D, S}, param) where {D, S<:CompositionalSystem}
    error()
end

@terv_secondary function update_as_secondary!(rho, m::TwoPhaseCompositionalDensities, model::SimulationModel{D, S}, param, FlashResults) where {D, S<:CompositionalSystem}
    error()
end


