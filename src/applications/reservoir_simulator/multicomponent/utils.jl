
get_components(sys::MultiComponentSystem) = sys.components
number_of_components(sys::MultiComponentSystem) = length(get_components(sys))

liquid_phase_index(sys::MultiPhaseCompositionalSystemLV) = phase_index(sys, LiquidPhase())
vapor_phase_index(sys::MultiPhaseCompositionalSystemLV) = phase_index(sys, VaporPhase())
other_phase_index(sys::MultiPhaseCompositionalSystemLV{E, T, O}) where {E, T, O<:Nothing} = error("Other phase not present")
other_phase_index(sys::MultiPhaseCompositionalSystemLV{E, T, O}) where {E, T, O} = phase_index(sys, O())

phase_index(sys, phase) = findfirst(isequal(phase), sys.phases)
has_other_phase(sys) = true
has_other_phase(sys::MultiPhaseCompositionalSystemLV{E, T, O}) where {E, T, O<:Nothing} = false