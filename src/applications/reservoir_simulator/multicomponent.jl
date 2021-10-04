using MultiComponentFlash
export TwoPhaseCompositionalSystem

abstract type MultiComponentSystem <: MultiPhaseSystem end
abstract type CompositionalSystem <: MultiComponentSystem end

struct TwoPhaseCompositionalSystem <: CompositionalSystem
    phases
    components
    equation_of_state
    flash_method
    function TwoPhaseCompositionalSystem(phases, equation_of_state; flash_method = SSIFlash())
        c = equation_of_state.component_names
        new(phases, c, equation_of_state, flash_method)
    end
end

get_components(sys::MultiComponentSystem) = sys.components
number_of_components(sys::MultiComponentSystem) = length(get_components(sys))