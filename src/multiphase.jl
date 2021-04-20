export MultiPhaseSystem, ImmiscibleMultiPhaseSystem, SinglePhaseSystem
export LiquidPhase, VaporPhase
export number_of_phases, get_short_name, get_name

export allocate_storage
# Abstract multiphase system
abstract type MultiPhaseSystem <: TervSystem end


function get_phases(sys::MultiPhaseSystem)
    return sys.phases
end

function number_of_phases(sys::MultiPhaseSystem)
    return length(get_phases(sys))
end

## 
function allocate_storage(G, sys)
    d = Dict()
    allocate_storage!(d, G, sys)
    return d
end

function allocate_storage!(d, G, sys::MultiPhaseSystem)
    nph = number_of_phases(sys)
    phases = get_phases(sys)
    npartials = nph
    nc = number_of_cells(G)
    nhf = number_of_half_faces(G)
    for ph in phases
        sname = get_short_name(ph)
        law = setup_conservationlaw(G, npartials)
        d[string("ConservationLaw_", sname)] = law
        d[string("Mobility_", sname)] = allocate_vector_ad(nc, npartials)
        d[string("Accmulation_", sname)] = allocate_vector_ad(nc, npartials)
        d[string("Flux_", sname)] = allocate_vector_ad(nhf, npartials)
    end
    jac = get_incomp_matrix(G)
    d["Jacobian"] = repeat(jac, nph, nph)
    d["Residual"] = zeros(nc*nph)
end

## Systems
# Immiscible multiphase system
struct ImmiscibleSystem <: MultiPhaseSystem
    phases::AbstractVector
end


# Single-phase
struct SinglePhaseSystem <: MultiPhaseSystem
    phase
end

function get_phases(sys::SinglePhaseSystem)
    return [sys.phase]
end

function number_of_phases(::SinglePhaseSystem)
    return 1
end

## Phases
# Abstract phase
abstract type AbstractPhase end

function get_short_name(phase::AbstractPhase)
    return get_name(phase)[1:1]
end

# Liquid phase
struct LiquidPhase <: AbstractPhase end

function get_name(::LiquidPhase)
    return "Liquid"
end

# Vapor phases
struct VaporPhase <: AbstractPhase end

function get_name(::VaporPhase)
    return "Vapor"
end

