export MultiPhaseSystem, ImmiscibleMultiPhaseSystem, SinglePhaseSystem
export LiquidPhase, VaporPhase
export number_of_phases, get_short_name, get_name, subscript
export update_linearized_system!
export SourceTerm
export setup_state, setup_state!

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

function subscript(prefix::String, phase::AbstractPhase)
    return string(prefix, "_", get_short_name(phase))
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

## Main implementation
function setup_state(model, arg...)
    d = Dict{String, Any}()
    setup_state!(d, model, model.grid, model.system, arg...)
    return d
end

function setup_state!(d, model, G, sys::MultiPhaseSystem, pressure::Union{Real, AbstractVector})
    nc = number_of_cells(G)
    if isa(pressure, AbstractVector)
        p = deepcopy(pressure)
        @assert length(pressure) == nc
    else
        p = repeat([pressure], nc)
    end
    p = transfer(model.context, p)
    d["Pressure"] = p
    phases = get_phases(sys)
    for phase in phases
        d[subscript("PhaseMass", phase)] = similar(p)
    end
end

function update_state!(model, storage)
    lsys = storage["LinearizedSystem"]
    storage["state"]["Pressure"] += lsys.dx
end


function convert_state_ad(model, state)
    context = model.context
    stateAD = deepcopy(state)
    vars = String.(keys(state))

    primary = get_primary_variable_names(model)
    n_partials = length(primary)
    # Loop over primary variables and set them to AD, with ones at the correct diagonal
    for i in 1:n_partials
        p = primary[i]
        stateAD[p] = allocate_vector_ad(stateAD[p], n_partials, diag_pos = i, context = context)
    end
    secondary = setdiff(vars, primary)
    # Loop over secondary variables and initialize as AD with zero partials
    for s in secondary
        stateAD[s] = allocate_vector_ad(stateAD[s], n_partials, context = context)
    end
    return stateAD
end

function number_of_primary_variables(model)
    return length(get_primary_variable_names(model))
end

function get_primary_variable_names(model)
    return get_primary_variable_names(model, model.formulation.primary_variables)
end

function get_primary_variable_names(model::SimulationModel{G, S}, ::DefaultPrimaryVariables) where {G<:Any, S<:SinglePhaseSystem}
    return ["Pressure"]
end

function allocate_storage!(d, model::SimulationModel{T, S}) where {T<:Any, S<:MultiPhaseSystem}
    G = model.grid
    sys = model.system
    context = model.context

    nph = number_of_phases(sys)
    phases = get_phases(sys)
    npartials = nph
    nc = number_of_cells(G)
    nhf = number_of_half_faces(G)

    A_p = get_sparsity_pattern(G)
    if nph == 1
        jac = A_p
    else
        jac = repeat(A_p, nph, nph) # This is super slow even with nph = 1?!
    end

    n_dof = nc*nph
    dx = zeros(n_dof)
    r = zeros(n_dof)
    lsys = LinearizedSystem(jac, r, dx)
    alloc = (n) -> allocate_vector_ad(n, npartials, context = context)
    alloc_value = (n) -> allocate_vector_ad(n, 0, context = context)
    for phaseNo in eachindex(phases)
        ph = phases[phaseNo]
        sname = get_short_name(ph)
        law = ConservationLaw(G, lsys, npartials, context = context)
        d[subscript("ConservationLaw", ph)] = law
        # Mobility of phase
        d[subscript("Mobility", ph)] = alloc(nc)
        # Mass density of phase
        d[subscript("Density", ph)] = alloc(nc)
        # Mobility * Density. We compute store this separately since density
        # and mobility are both used in other spots
        d[subscript("MassMobility", ph)] = alloc(nc)
        d[subscript("TotalMass", ph)] = alloc(nc)
        d[subscript("TotalMass0", ph)] = alloc_value(nc)
    end
    # Transfer linearized system afterwards since the above manipulations are
    # easier to do on CPU
    d["LinearizedSystem"] = transfer(context, lsys)
end

function initialize_storage!(d, model::SimulationModel{T, S}) where {T<:Any, S<:MultiPhaseSystem}
    update_properties!(model, d)
    for ph in get_phases(model.system)
        m = d[subscript("TotalMass", ph)]
        m0 = d[subscript("TotalMass0", ph)]
        @. m0 = value(m)
    end
end

function update_equations!(model::SimulationModel{G, S}, storage; 
    dt = nothing, sources = nothing) where {G<:MinimalTPFAGrid, S<:MultiPhaseSystem}
    update_properties!(model, storage)
    phases = get_phases(model.system)
    for phNo in eachindex(phases)
        phase = phases[phNo]
        law = storage[subscript("ConservationLaw", phase)]
        update_accumulation!(model, storage, phase, dt)
        update_half_face_flux!(model, storage, phase)

        if !isnothing(sources)
            # @debug "Inserting source terms."
            insert_sources(law.accumulation, sources, phNo)
        end
    end
end

function update_properties!(model, storage)
    for phase in get_phases(model.system)
        # Update a few values
        update_density!(model, storage, phase)
        update_mobility!(model, storage, phase)
        update_mass_mobility!(model, storage, phase)
        update_total_mass!(model, storage, phase)
    end
end

# Updates of various cell properties follows
function update_mobility!(model::SimulationModel{G, S}, storage, phase::AbstractPhase) where {G<:Any, S<:SinglePhaseSystem}
    mob = storage[subscript("Mobility", phase)]
    mu = storage["parameters"][subscript("Viscosity", phase)]
    fapply!(mob, () -> 1/mu)
end

function update_density!(model, storage, phase::AbstractPhase)
    param = storage["parameters"]
    state = storage["state"]
    p = state["Pressure"]
    
    d = subscript("Density", phase)
    rho = storage[d]
    r = param[d]
    if isa(r, NamedTuple)
        f_rho = (p) -> r.rhoS*exp((p - r.pRef)*r.c)
    else
        # Function handle
        f_rho = r
    end
    fapply!(rho, f_rho, p)
    return rho
end

function update_total_mass!(model::SimulationModel{G, S}, storage, phase::AbstractPhase) where {G<:Any, S<:SinglePhaseSystem}
    pv = model.grid.pv
    rho = storage[subscript("Density", phase)]
    totMass = storage[subscript("TotalMass", phase)]
    fapply!(totMass, *, rho, pv)
end

function update_mass_mobility!(model, storage, phase::AbstractPhase)
    mobrho = storage[subscript("MassMobility", phase)]
    mob = storage[subscript("Mobility", phase)]
    rho = storage[subscript("Density", phase)]
    # Assign the values
    fapply!(mobrho, *, mob, rho)
end

# Update of discretization terms
function update_accumulation!(model, storage, phase::AbstractPhase, dt)
    law = storage[subscript("ConservationLaw", phase)]
    mass = storage[subscript("TotalMass", phase)]
    mass0 = storage[subscript("TotalMass0", phase)]
    fapply!(law.accumulation, (m, m0) -> (m - m0)/dt, mass, mass0)
end

function update_half_face_flux!(model, storage, phase::AbstractPhase)
    p = storage["state"]["Pressure"]
    law = storage[subscript("ConservationLaw", phase)]
    mmob = storage[subscript("MassMobility", phase)]

    half_face_flux!(law.half_face_flux, mmob, p, model.grid)
end

# Source terms, etc
@inline function insert_sources(acc, sources, phNo)
    for src in sources
        @inbounds acc[src.cell] -= src.values[phNo]
    end
end

@inline function insert_sources(acc::CuArray, sources, phNo)
    s = cu(map(x -> x.values[phNo], sources))
    i = cu(map(x -> x.cell, sources))
    @. acc[i] -= s
end

# Updating of linearized system
function update_linearized_system!(model::TervModel, storage)
    sys = model.system;
    sys::MultiPhaseSystem

    lsys = storage["LinearizedSystem"]
    phases = get_phases(sys)
    for phase in phases
        sname = get_short_name(phase)
        law = storage[subscript("ConservationLaw", phase)]
        update_linearized_system!(model.grid, lsys, law)
    end
end

function update_linearized_system!(G, lsys::LinearizedSystem, law::ConservationLaw)
    apos = law.accumulation_jac_pos
    jac = lsys.jac
    r = lsys.r
    # Fill in diagonal
    fill_accumulation!(jac, r, law.accumulation, apos)
    # Fill in off-diagonal
    fpos = law.half_face_flux_jac_pos
    fill_half_face_fluxes(jac, r, G.conn_pos, G.conn_data, law.half_face_flux, apos, fpos)
end

"Fill acculation term onto diagonal with pre-determined access pattern into jac"
function fill_accumulation!(jac, r, acc, apos)
    @inbounds Threads.@threads for col = 1:size(apos, 2)
        r[col] = acc[col].value
        fill_accumulation_jac!(jac.nzval, acc, apos, col)
    end
end

function fill_accumulation_jac!(nzval, acc, apos, col)
    @inbounds for derNo = 1:size(apos, 1)
        index = apos[derNo, col]
        nzval[index] = acc[col].partials[derNo]
    end
end
"Fill fluxes onto diagonal with pre-determined access pattern into jac. Essentially performs Div ( flux )"
function fill_half_face_fluxes(jac, r, conn_pos, conn_data, half_face_flux, apos, fpos)
    Threads.@threads for col = 1:length(apos)
        @inbounds for i = conn_pos[col]:(conn_pos[col+1]-1)
            @inbounds r[col] += half_face_flux[i].value
            fill_half_face_fluxes_jac!(half_face_flux, jac.nzval, apos, fpos, col, i)
        end
    end
end

function fill_half_face_fluxes_jac!(half_face_flux, nzval, apos, fpos, col, i)
    @inbounds for derNo = 1:size(apos, 1)
        index = fpos[derNo, i]
        diag_index = apos[derNo, col]
        df_di = half_face_flux[i].partials[derNo]
        nzval[index] = -df_di
        nzval[diag_index] += df_di
    end
end