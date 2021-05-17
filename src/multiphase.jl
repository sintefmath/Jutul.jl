export MultiPhaseSystem, ImmiscibleSystem, SinglePhaseSystem
export LiquidPhase, VaporPhase
export number_of_phases, get_short_name, get_name, subscript
export update_linearized_system!
export SourceTerm, build_forces
export setup_state, setup_state!

export allocate_storage, update_equations!

using CUDA
# Abstract multiphase system
abstract type MultiPhaseSystem <: TervSystem end


function get_phases(sys::MultiPhaseSystem)
    return sys.phases
end


function number_of_phases(sys::MultiPhaseSystem)
    return length(get_phases(sys))
end

struct SourceTerm
    cell
    value
    fractional_flow
    composition
end

function SourceTerm(cell, value; fractional_flow = [1.0], compositions = nothing)
    @assert sum(fractional_flow) == 1.0 "Fractional flow for source term in cell $cell must sum to 1."
    cell::Integer
    value::AbstractFloat
    SourceTerm(cell, value, fractional_flow, compositions)
end

function build_forces(model::SimulationModel{G, S}; sources = nothing) where {G<:Any, S<:MultiPhaseSystem}
    return (sources = sources,)
end

## Systems
# Immiscible multiphase system
struct ImmiscibleSystem <: MultiPhaseSystem
    phases::AbstractVector
end

# function ImmiscibleSystem(phases)
#    @assert length(phases) > 1 "System should have at least two phases. For single-phase, use SinglePhaseSystem instead."
# end

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
function add_extra_state_fields!(state, model::SimulationModel{G, S}) where {G<:Any, S<:MultiPhaseSystem}
    nc = number_of_cells(model.domain)
    nph = number_of_phases(model.system)
    state[:TotalMass] = transfer(model.context, zeros(nph, nc))
end

# Primary variable logic

# Pressure as primary variable
struct Pressure <: ScalarPrimaryVariable
    symbol
end

function Pressure()
    Pressure(:Pressure)
end
# Saturations as primary variable
struct Saturations <: GroupedPrimaryVariables
    symbol
    phases
    dsMax
end

function Saturations(phases::AbstractArray, dsMax = 0.2)
    Saturations(:Saturations, phases, dsMax)
end


function degrees_of_freedom_per_unit(v::Saturations)
    return length(v.phases) - 1
end

#function get_names(v::Saturations)
#    return map((x) -> subscript(v.name, x), v.phases[1:end-1])
# end

@inline function maximum_value(::Saturations) 1 end
@inline function minimum_value(::Saturations) 0 end
@inline function absolute_increment_limit(p::Saturations) p.dsMax end

function initialize_primary_variable_ad(state, model, pvar::Saturations, offset, npartials)
    name = get_symbol(pvar)
    nph = length(pvar.phases)
    # nph - 1 primary variables, with the last saturation being initially zero AD
    dp = vcat((1:nph-1) .+ offset, 0)
    v = state[name]
    v = allocate_array_ad(v, diag_pos = dp, context = model.context, npartials = npartials)
    for i in 1:size(v, 2)
        v[end, i] = 1 - sum(v[1:end-1, i])
    end
    state[name] = v
    return state
end

function initialize_primary_variable_value(state, model, pvar::Saturations, val)
    name = get_symbol(pvar)
    if isa(val, Dict)
        val = val[name]
    end
    val::AbstractVecOrMat # Should be a vector or a matrix
    V = deepcopy(val)
    @assert size(V, 1) == number_of_phases(model.system)
    n = number_of_units(model, pvar)
    if isa(V, AbstractVector)
        V = repeat(V, 1, n)
    end
    @assert size(V, 2) == n
    state[name] = transfer(model.context, V)
    return state
end

function update_state!(state, p::Saturations, model, dx)
    nu = number_of_units(model, p)
    nph = degrees_of_freedom_per_unit(p) + 1
    
    abs_max = absolute_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)

    s = state[get_symbol(p)]
    Threads.@threads for cell = 1:nu
        dlast = 0
        @inbounds for ph = 1:(nph-1)
            v = value(s[ph, cell])
            dv = dx[cell + (ph-1)*nu]
            dv = choose_increment(v, dv, abs_max, nothing, minval, maxval)
            dlast -= dv
            s[ph, cell] += dv
        end
        s[nph, cell] += dlast
    end
end

function select_primary_variables(domain, system::SinglePhaseSystem, formulation)
    return [Pressure()]
end

function select_primary_variables(domain, system::ImmiscibleSystem, formulation)
    return [Pressure(), Saturations(system.phases)]
end

function allocate_properties!(props, storage, model::SimulationModel{T, S}) where {T<:Any, S<:MultiPhaseSystem}
    G = model.domain
    sys = model.system
    context = model.context

    nph = number_of_phases(sys)
    npartials = nph
    nc = number_of_cells(G)
    alloc = (n) -> allocate_array_ad(nph, n, context = context, npartials = npartials)

    # Mobility of phase
    props[:Mobility] = alloc(nc)
    # Mass density of phase
    props[:Density] = alloc(nc)
    # Mobility * Density. We compute store this separately since density
    # and mobility are both used in other spots
    props[:MassMobility] = alloc(nc)
end

#function allocate_linearized_system!(d, model::SimulationModel{T, S}) where {T<:Any, S<:MultiPhaseSystem}
#    @debug "Allocating lsys mphase"
#
#    G = model.domain
#    nph = number_of_phases(model.system)
#    nc = number_of_cells(G)
#
#    jac = get_sparsity_pattern(G, nph, nph)
#
#    n_dof = nc*nph
#    dx = zeros(n_dof)
#    r = zeros(n_dof)
#    lsys = LinearizedSystem(jac, r, dx)
#    d["LinearizedSystem"] = transfer(model.context, lsys)
#    return lsys
# end

function allocate_equations!(eqs, storage, model::SimulationModel{T, S}) where {T<:Any, S<:MultiPhaseSystem}
    nph = number_of_phases(model.system)
    npartials = nph
    law = ConservationLaw(model.domain, npartials, context = model.context, equations_per_unit = nph)
    eqs[:MassConservation] = law
    return eqs
end

function initialize_storage!(storage, model::SimulationModel{T, S}) where {T<:Any, S<:MultiPhaseSystem}
    update_properties!(storage, model)
    m = storage.state.TotalMass
    m0 = storage.state0.TotalMass
    @. m0 = value(m)
    if isa(model.context, SingleCUDAContext)
        # No idea why this is needed, but it is.
        CUDA.synchronize()
    end
end

function update_properties!(storage, model::SimulationModel{G, S}) where {G<:Any, S<:MultiPhaseSystem}
    update_density!(storage, model)
    update_mobility!(storage, model)
    update_mass_mobility!(storage, model)
    update_total_mass!(storage, model)
end

# Updates of various cell properties follows
function update_mobility!(storage, model::SimulationModel{G, S}) where {G<:Any, S<:SinglePhaseSystem}
    mob = storage.properties.Mobility
    mu = storage.parameters.Viscosity[1]
    fapply!(mob, () -> 1/mu)
end

function update_mobility!(storage, model::SimulationModel{G, S}) where {G<:Any, S<:ImmiscibleSystem}
    p = storage.parameters
    mob = storage.properties.Mobility
    n = p.CoreyExponents
    mu = p.Viscosity

    s = storage.state.Saturations
    @. mob = s^n/mu
end

function update_density!(storage, model)
    rho_input = storage.parameters.Density
    state = storage.state
    p = state.Pressure
    rho = storage.properties.Density
    for i in 1:number_of_phases(model.system)
        rho_i = view(rho, i, :)
        r = rho_input[i]
        if isa(r, NamedTuple)
            f_rho = (p) -> r.rhoS*exp((p - r.pRef)*r.c)
        else
            # Function handle
            f_rho = r
        end
        fapply!(rho_i, f_rho, p)
    end
    return rho
end

function get_pore_volume(model) model.domain.grid.pore_volumes' end

function update_total_mass!(storage, model::SimulationModel{G, S}) where {G<:Any, S<:SinglePhaseSystem}
    pv = get_pore_volume(model)
    rho = storage.properties.Density
    totMass = storage.state.TotalMass
    fapply!(totMass, *, rho, pv)
end

function update_total_mass!(storage, model::SimulationModel{G, S}) where {G<:Any, S<:ImmiscibleSystem}
    state = storage.state
    pv = get_pore_volume(model)
    rho = storage.properties.Density
    totMass = state.TotalMass
    s = state.Saturations

    @. totMass = rho*pv*s
    # fapply!(totMass, *, rho, pv, s)
end

function update_mass_mobility!(storage, model)
    props = storage.properties
    mobrho = props.MassMobility
    mob = props.Mobility
    rho = props.Density
    # Assign the values
    fapply!(mobrho, *, mob, rho)
end

# Update of discretization terms
function update_accumulation!(law, storage, model, dt)
    state = storage.state
    state0 = storage.state0
    acc = law.accumulation
    mass = state.TotalMass
    mass0 = state0.TotalMass
    fapply!(acc, (m, m0) -> (m - m0)/dt, mass, mass0)
    return acc
end

function update_half_face_flux!(law, storage, model)
    p = storage.state.Pressure
    mmob = storage.properties.MassMobility

    half_face_flux!(model, law.half_face_flux, mmob, p)
end

function apply_forces_to_equation!(storage, model::SimulationModel{D, S}, eq::ConservationLaw, force::Vector{SourceTerm}) where {D<:Any, S<:MultiPhaseSystem}
    @debug "Applying source terms"
    acc = eq.accumulation
    mob = storage.properties.Mobility
    insert_phase_sources(mob, acc, force)
end

function insert_phase_sources(mob, acc, sources)
    nph = size(acc, 1)
    for src in sources
        v = src.value
        c = src.cell
        mobt = sum(mob[:, c])
        if v > 0
            for index in 1:nph
                f = src.fractional_flow[index]
                @inbounds acc[index, c] -= v*f
            end
        else
            for index in 1:nph
                f = mob[index, c]/mobt
                @inbounds acc[index, c] -= v*f
            end
        end
    end
end

@inline function insert_phase_sources(mob::CuArray, acc::CuArray, sources)
    @assert false "Not updated."
    s = cu(map(x -> x.values[phNo], sources))
    i = cu(map(x -> x.cell, sources))
    @. acc[i] -= s
end

# Accumulation: Base implementation
"Fill acculation term onto diagonal with pre-determined access pattern into jac"
function fill_accumulation!(jac, r, acc, apos, neq, context, ::KernelDisallowed)
    nzval = get_nzval(jac)
    nc = size(apos, 2)
    dim = size(apos, 1)
    nder = dim ÷ neq
    @inbounds Threads.@threads for cell = 1:nc
        for eq = 1:neq
            r[cell + (eq-1)*nc] = acc[eq, cell].value
        end
        fill_accumulation_jac!(nzval, acc, apos, cell, nder, neq)
    end
end

@inline function fill_accumulation_jac!(nzval, acc, apos, cell, nder, neq)
    for eqNo = 1:neq
        a = acc[eqNo, cell]
        for derNo = 1:nder
            nzval[apos[derNo + (eqNo-1)*nder, cell]] = a.partials[derNo]
        end
    end
end
# Kernel / CUDA version follows
function fill_accumulation!(jac, r, acc, apos, neq, context, ::KernelAllowed)
    @assert false "Not updated"
    @. r = value(acc)
    jz = get_nzval(jac)
    @inbounds for i = 1:size(apos, 1)
        jz[apos[i, :]] = map((x) -> x.partials[i], acc)
    end
    # kernel = fill_accumulation_jac_kernel(context.device, context.block_size)
    # event = kernel(jz, acc, apos, ndrange = size(apos))
    # wait(event)
end

"Kernel for filling Jacobian from accumulation term"
@kernel function fill_accumulation_jac_kernel(nzval, @Const(acc), @Const(apos))
    derNo, col = @index(Global, NTuple)
    @inbounds nzval[apos[derNo, col]] = acc[col].partials[derNo]
    # i = @index(Global, Linear)
    # @inbounds nzval[apos[1, i]] = acc[i].partials[1]
end

# Fluxes: Base implementation
"Fill fluxes onto diagonal with pre-determined access pattern into jac. Essentially performs Div ( flux )"
function fill_half_face_fluxes(jac, r, conn_pos, half_face_flux, apos, fpos, neq, context, ::KernelDisallowed)
    Jz = get_nzval(jac)
    nc = size(apos, 2)
    n = size(apos, 1)
    nder = n ÷ neq
    # Threads.@threads for cell = 1:nc
    for cell = 1:nc
        for i = conn_pos[cell]:(conn_pos[cell+1]-1)
            for eqNo = 1:neq
                # Update diagonal value
                f = half_face_flux[eqNo, i]
                r[cell + (eqNo-1)*nc] += f.value
                # Fill inn Jacobian values
                for derNo = 1:nder
                    i_pos = derNo + (eqNo-1)*nder

                    index = fpos[i_pos, i]
                    diag_index = apos[i_pos, cell]
                    df_di = f.partials[derNo]
                    Jz[index] = -df_di
                    Jz[diag_index] += df_di
                end
            end
        end
    end
end

# Kernel / CUDA version follows
function fill_half_face_fluxes(jac, r, conn_pos, half_face_flux, apos, fpos, context, ::KernelAllowed)
    d = size(apos)
    Jz = get_nzval(jac)

    # println(":: HF, value")
    begin
        ncol = d[2]
        kernel = fill_half_face_fluxes_val_kernel(context.device, context.block_size, ncol)
        event_val = kernel(r, half_face_flux, conn_pos, ndrange = ncol)
        # wait(event_val)
    end

    # println(":: HF, offdiag")
    if true
        begin
            @inbounds for i = 1:size(fpos, 1)
                Jz[fpos[i, :]] = map((x) -> -x.partials[i], half_face_flux)
            end
        end
        # println(":: HF, diag")
        begin
            kernel = fill_half_face_fluxes_jac_diag_kernel(context.device, context.block_size, d)
            event_jac = kernel(Jz, half_face_flux, apos, conn_pos, ndrange = d)
            # wait(event_jac)
        end
        wait(event_val)
        wait(event_jac)
    else
        println(":: HF, full kernel")
        @time begin
            kernel = fill_half_face_fluxes_jac_kernel(context.device, context.block_size, d)
            event_jac = kernel(Jz, half_face_flux, apos, fpos, conn_pos, ndrange = d)
            wait(event_jac)
        end
    end
end

@kernel function fill_half_face_fluxes_val_kernel(r, @Const(half_face_flux), @Const(conn_pos))
    col = @index(Global, Linear)
    v = 0
    @inbounds for i = conn_pos[col]:(conn_pos[col+1]-1)
        # Update diagonal value
        v += half_face_flux[i].value
    end
    @inbounds r[col] += v
end


@kernel function fill_half_face_fluxes_jac_diag_kernel(nzval, @Const(half_face_flux), @Const(apos), @Const(conn_pos))
    derNo, col = @index(Global, NTuple)
    v = 0
    @inbounds for i = conn_pos[col]:(conn_pos[col+1]-1)
        v += half_face_flux[i].partials[derNo]
    end
    @inbounds nzval[apos[derNo, col]] += v
    nothing
end

@kernel function fill_half_face_fluxes_jac_kernel(nzval, @Const(half_face_flux), @Const(apos), @Const(fpos), @Const(conn_pos))
    derNo, col = @index(Global, NTuple)
    v = 0
    @inbounds for i = conn_pos[col]:(conn_pos[col+1]-1)
        index = fpos[derNo, i]
        df_di = half_face_flux[i].partials[derNo]
        nzval[index] = -df_di
        v += df_di
    end
    @inbounds diag_index = apos[derNo, col]
    @inbounds nzval[diag_index] += v # Should this be atomic?
    nothing
end
