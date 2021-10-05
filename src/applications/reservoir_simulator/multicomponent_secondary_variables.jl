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
end

default_value(model, ::FlashResults) = FlashedMixture2Phase(model.system.equation_of_state)

function initialize_variable_value!(state, model, pvar::FlashResults, symb, val::AbstractDict; need_value = false)
    @assert need_value == false
    v = default_value(model, pvar)
    V = repeat([v], number_of_entities(model, pvar))
    initialize_variable_value!(state, model, pvar, symb, V)
end

function select_secondary_variables_system!(S, domain, system::CompositionalSystem, formulation)
    nph = number_of_phases(system)
    S[:PhaseMassDensities] = ConstantCompressibilityDensities(nph)
    S[:TotalMasses] = TotalMasses()
    S[:FlashResults] = FlashResults()
    S[:Saturations] = Saturations()
    S[:PhaseViscosities] = ConstantVariables(1e-3*ones(nph)) # 1 cP for all phases by default
end

degrees_of_freedom_per_entity(model, v::MassMobilities) = number_of_phases(model.system)*number_of_components(model.system)