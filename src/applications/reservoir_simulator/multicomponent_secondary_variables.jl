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