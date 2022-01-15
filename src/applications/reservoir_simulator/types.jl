
abstract type MultiPhaseSystem <: JutulSystem end
abstract type MultiComponentSystem <: MultiPhaseSystem end
abstract type CompositionalSystem <: MultiComponentSystem end

struct MultiPhaseCompositionalSystemLV{E, T, O} <: CompositionalSystem where T<:Tuple
    phases::T
    components
    equation_of_state::E
    function MultiPhaseCompositionalSystemLV(equation_of_state, phases = (LiquidPhase(), VaporPhase()); other_name = "Water")
        c = copy(equation_of_state.mixture.component_names)
        phases = tuple(phases...)
        T = typeof(phases)
        nph = length(phases)
        @assert nph == 2 || nph == 3
        if nph == 3
            other = only(filter(x -> !(isa(x, LiquidPhase) || isa(x, VaporPhase)), phases))
            O = typeof(other)
            push!(c, other_name)
        else
            O = Nothing
        end
        only(findall(isequal(LiquidPhase()), phases))
        only(findall(isequal(VaporPhase()), phases))
        new{typeof(equation_of_state), T, O}(phases, c, equation_of_state)
    end
end

struct ImmiscibleSystem{T} <: MultiPhaseSystem where T<:Tuple
    phases::T
end

ImmiscibleSystem(phases::AbstractVector) = ImmiscibleSystem(tuple(phases...))

struct SinglePhaseSystem <: MultiPhaseSystem
    phase
end
