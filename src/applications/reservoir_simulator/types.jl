
abstract type MultiPhaseSystem <: TervSystem end
abstract type MultiComponentSystem <: MultiPhaseSystem end
abstract type CompositionalSystem <: MultiComponentSystem end

struct TwoPhaseCompositionalSystem <: CompositionalSystem
    phases
    components
    equation_of_state
    function TwoPhaseCompositionalSystem(phases, equation_of_state)
        c = equation_of_state.mixture.component_names
        new(phases, c, equation_of_state)
    end
end
