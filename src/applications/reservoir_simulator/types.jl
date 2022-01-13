
abstract type MultiPhaseSystem <: TervSystem end
abstract type MultiComponentSystem <: MultiPhaseSystem end
abstract type CompositionalSystem <: MultiComponentSystem end

struct TwoPhaseCompositionalSystem{E} <: CompositionalSystem
    phases
    components
    equation_of_state::E
    function TwoPhaseCompositionalSystem(phases, equation_of_state)
        c = equation_of_state.mixture.component_names
        new{typeof(equation_of_state)}(phases, c, equation_of_state)#, length(c), length(phases))
    end
end
