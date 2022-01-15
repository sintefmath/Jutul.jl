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

"""
A single saturation that represents the "other" phase in a
three phase compositional system where two phases are predicted by an EoS
"""
Base.@kwdef struct ImmiscibleSaturation <: ScalarVariable
    ds_max = 0.2
end

maximum_value(::ImmiscibleSaturation) = 1.0 - MINIMUM_COMPOSITIONAL_SATURATION
minimum_value(::ImmiscibleSaturation) = 0.0
relative_increment_limit(s::ImmiscibleSaturation) = s.ds_max