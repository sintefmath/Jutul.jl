export MultiPhaseSystem, ImmiscibleMultiPhaseSystem, SinglePhaseSystem
export LiquidPhase, VaporPhase
export number_of_phases, get_short_name, get_name
export update_linearized_system!
export SourceTerm

export allocate_storage, update_equations!
# Abstract multiphase system
abstract type MultiPhaseSystem <: TervSystem end


function get_phases(sys::MultiPhaseSystem)
    return sys.phases
end

function number_of_phases(sys::MultiPhaseSystem)
    return length(get_phases(sys))
end

struct SourceTerm{R<:Real,I<:Integer}
    cell::I
    values::AbstractVector{R}
end
## 
#function init_state(model)
#    d = dict()
#    init_state!(d, model)
#    return d
#end
#
#function init_state!(model)
#    sys = model.system
#    sys::MultiPhaseSystem
#    d = dict()
#    init_state!(d, model)
#end


function allocate_storage!(d, G, sys::MultiPhaseSystem)
    nph = number_of_phases(sys)
    phases = get_phases(sys)
    npartials = nph
    nc = number_of_cells(G)
    nhf = number_of_half_faces(G)

    A_p = get_incomp_matrix(G)
    jac = repeat(A_p, nph, nph)

    n_dof = nc*nph
    dx = zeros(n_dof)
    r = zeros(n_dof)
    lsys = LinearizedSystem(jac, r, dx)
    d["LinearizedSystem"] = lsys
    # hasAcc = !isa(sys, SinglePhaseSystem) # and also incompressible!!
    for phaseNo in eachindex(phases)
        ph = phases[phaseNo]
        sname = get_short_name(ph)
        law = ConservationLaw(G, lsys, npartials)
        d[string("ConservationLaw_", sname)] = law
        d[string("Mobility_", sname)] = allocate_vector_ad(nc, npartials)
        # d[string("Accmulation_", sname)] = allocate_vector_ad(nc, npartials)
        # d[string("Flux_", sname)] = allocate_vector_ad(nhf, npartials)
    end
end

function update_equations!(model, storage; dt = nothing, sources = nothing)
    sys = model.system;
    sys::MultiPhaseSystem

    state = storage["state"]
    state0 = storage["state0"]
    

    p = state["Pressure"]
    p0 = state0["Pressure"]
    pv = model.G.pv

    phases = get_phases(sys)
    for phNo in eachindex(phases)
        phase = phases[phNo]
        sname = get_short_name(phase)
        # Parameters - fluid properties
        rho = storage["parameters"][string("Density_", sname)]
        mu = storage["parameters"][string("Viscosity_", sname)]
        # Storage structure
        law = storage[string("ConservationLaw_", sname)]
        mob = storage[string("Mobility_", sname)]
        mob .= 1/mu

        half_face_flux!(law.half_face_flux, mob, p, model.G)
        law.accumulation .= pv.*(rho.(p) - rho.(p0))./dt
        if !isnothing(sources)
            for src in sources
                law.accumulation[src.cell] += src.values[phNo]
            end
        end
    end
end

function update_linearized_system!(model::TervModel, storage)
    sys = model.system;
    sys::MultiPhaseSystem

    lsys = storage["LinearizedSystem"]
    phases = get_phases(sys)
    for phase in phases
        sname = get_short_name(phase)
        law = storage[string("ConservationLaw_", sname)]
        update_linearized_system!(model.G, lsys, law)
    end
end

function update_linearized_system!(G, lsys::LinearizedSystem, law::ConservationLaw)
    apos = law.accumulation_jac_pos
    jac = lsys.jac
    r = lsys.r
    # Fill in diagonal
    for i = 1:size(apos, 2)
        r[i] = law.accumulation[i].value
        for derNo = 1:size(apos, 1)
            index = apos[derNo, i]

            jac.nzval[index] = law.accumulation[i].partials[derNo]
        end
    end
    # Fill in off-diagonal
    fpos = law.half_face_flux_jac_pos
    for i = 1:size(fpos, 2)
        cell_index = G.conn_data[i].self
        r[cell_index] += law.half_face_flux[i].value
        for derNo = 1:size(apos, 1)
            index = fpos[derNo, i]
            diag_index = apos[derNo, cell_index]
            df_di = law.half_face_flux[i].partials[derNo]
            jac.nzval[index] = -df_di
            jac.nzval[diag_index] += df_di
        end
    end
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

