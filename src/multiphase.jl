export MultiPhaseSystem, ImmiscibleMultiPhase, SinglePhase
export number_of_phases

# Abstract multiphase system
abstract type MultiPhaseSystem <: TervSystem end

function number_of_phases(::MultiPhaseSystem)
    return nothing
end

function allocate_storage(::MultiPhaseSystem)
    return nothing
end

function allocate_storage(G, sys::MultiPhaseSystem)
    d = Dict()
    npartials = get_number_of_phases(sys)
    for ph in phases(sys)
        law = setup_conservationlaw(G, npartials)
        d[ph] = law
        d[]
    end
    return nothing
end

# Abstract phase
abstract type AbstractPhase

end

struct LiquidPhase <: AbstractPhase

end

struct VaporPhase <: AbstractPhase

end

# Immiscible multiphase system
struct ImmiscibleSystem <: MultiPhaseSystem
    phases
end


# Single-phase
struct SinglePhaseSystem <: MultiPhaseSystem
    phase
end

function number_of_phases(::SinglePhaseSystem)
    return 1
end
