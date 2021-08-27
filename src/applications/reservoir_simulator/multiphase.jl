export MultiPhaseSystem, ImmiscibleSystem, SinglePhaseSystem
export AqueousPhase, LiquidPhase, VaporPhase
export number_of_phases, get_short_name, get_name, subscript
export update_linearized_system!
export SourceTerm, build_forces
export setup_state, setup_state!

export setup_storage, update_equations!

export Pressure, Saturations, TotalMasses, TotalMass

using CUDA
# Abstract multiphase system
abstract type MultiPhaseSystem <: TervSystem end


function get_phases(sys::MultiPhaseSystem)
    return sys.phases
end

function number_of_phases(sys::MultiPhaseSystem)
    return length(get_phases(sys))
end

struct SourceTerm{I, F, T} <: TervForce
    cell::I
    value::F
    fractional_flow::T
    function SourceTerm(cell, value; fractional_flow = [1.0])
        @assert sum(fractional_flow) == 1.0 "Fractional flow for source term in cell $cell must sum to 1."
        f = Tuple(fractional_flow)
        return new{typeof(cell), typeof(value), typeof(f)}(cell, value, f)
    end
end

function cell(s::SourceTerm{I, T}) where {I, T} 
    return s.cell::I
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
    return get_name(phase)[]
end

function subscript(prefix::String, phase::AbstractPhase)
    return string(prefix, "_", get_short_name(phase))
end
# Aqueous phase
struct AqueousPhase <: AbstractPhase end

function get_name(::AqueousPhase)
    return "Aqueous"
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
# Primary variable logic

# Pressure as primary variable
struct Pressure <: ScalarVariable
    dpMaxAbs
    dpMaxRel
    scale
    function Pressure(dpMaxAbs = nothing, dpMaxRel = nothing, scale = 1e8)
        new(dpMaxAbs, dpMaxRel, scale)
    end
end

function variable_scale(p::Pressure)
    return p.scale
end

@inline function absolute_increment_limit(p::Pressure) p.dpMaxAbs end
@inline function relative_increment_limit(p::Pressure) p.dpMaxRel end

# Saturations as primary variable
struct Saturations <: GroupedVariables
    dsMax
    function Saturations(dsMax = 0.2)
        new(dsMax)
    end
end

function degrees_of_freedom_per_unit(model, v::Saturations)
    return values_per_unit(model, v) - 1
end

function values_per_unit(model, v::Saturations)
    number_of_phases(model.system)
end

@inline function maximum_value(::Saturations) 1.0 end
@inline function minimum_value(::Saturations) 0.0 end
@inline function absolute_increment_limit(p::Saturations) p.dsMax end

function initialize_primary_variable_ad!(state, model, pvar::Saturations, state_symbol, npartials; offset = 0, kwarg...)
    nph = values_per_unit(model, pvar)
    # nph - 1 primary variables, with the last saturation being initially zero AD
    dp = vcat((1:nph-1) .+ offset, 0)
    v = state[state_symbol]
    v = allocate_array_ad(v, diag_pos = dp, context = model.context, npartials = npartials; kwarg...)
    for i in 1:size(v, 2)
        v[end, i] = 1 - sum(v[1:end-1, i])
    end
    state[state_symbol] = v
    return state
end

function update_primary_variable!(state, p::Saturations, state_symbol, model, dx)
    nph, nu = value_dim(model, p)
    abs_max = absolute_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)

    s = state[state_symbol]
    Threads.@threads for cell = 1:nu
    # for cell = 1:nu
        dlast = 0
        @inbounds for ph = 1:(nph-1)
            v = value(s[ph, cell])
            dv = dx[cell + (ph-1)*nu]
            dv = choose_increment(v, dv, abs_max, nothing, minval, maxval)
            dlast -= dv
            s[ph, cell] += dv
        end
        @inbounds s[nph, cell] += dlast
    end
end

# Total component masses
struct TotalMasses <: GroupedVariables end

function degrees_of_freedom_per_unit(model::SimulationModel{G, S}, v::TotalMasses) where {G<:Any, S<:MultiPhaseSystem}
    number_of_phases(model.system)
end

@inline function minimum_value(::TotalMasses) 0 end

struct TotalMass <: ScalarVariable end
@inline function minimum_value(::TotalMass) 0 end


# Selection of variables
function select_primary_variables_system!(S, domain, system::SinglePhaseSystem, formulation)
    S[:Pressure] = Pressure()
end

function select_primary_variables_system!(S, domain, system::ImmiscibleSystem, formulation)
    S[:Pressure] = Pressure()
    S[:Saturations] = Saturations()
end

function select_equations_system!(eqs, domain, system::MultiPhaseSystem, formulation)
    nph = number_of_phases(system)
    eqs[:mass_conservation] = (ConservationLaw, nph)
end

function get_pore_volume(model)
    get_flow_volume(model.domain.grid)'
end

function get_flow_volume(grid::MinimalTPFAGrid)
    grid.pore_volumes
end

function get_flow_volume(grid)
    1
end

function apply_forces_to_equation!(storage, model::SimulationModel{D, S}, eq::ConservationLaw, force::V) where {V <: AbstractVector{SourceTerm{I, F, T}}, D, S<:MultiPhaseSystem} where {I, F, T}
    acc = get_diagonal_entries(eq)
    state = storage.state
    if haskey(state, :RelativePermeabilities)
        kr = state.RelativePermeabilities
    else
        kr = 1.0
    end
    mu = state.PhaseViscosities
    rhoS = get_reference_densities(model, storage)
    insert_phase_sources!(acc, kr, mu, rhoS, force)
end

function local_mobility(kr::Real, mu, ph, c)
    return kr./mu[ph, c]
end

function local_mobility(kr, mu, ph, c)
    return kr[ph, c]./mu[ph, c]
end

function phase_source(src, rhoS, kr, mu, ph)
    v = src.value
    c = src.cell
    if v > 0
        q = in_phase_source(src, v, c, kr, mu, ph)
    else
        q = out_phase_source(src, v, c, kr, mu, ph)
    end
    return rhoS*q
end


function in_phase_source(src, v, c, kr, mu, ph)
    f = src.fractional_flow[ph]
    return v*f
end

function out_phase_source(src, v, c, kr, mu, ph)
    mobT = 0
    mob = 0
    for i = 1:size(mu, 1)
        mi = local_mobility(kr, mu, i, c)
        mobT += mi
        if ph == i
            mob = mi
        end
    end
    f = mob/mobT
    return v*f
end

function insert_phase_sources!(acc, kr, mu, rhoS, sources)
    nph = size(acc, 1)
    for src in sources
        for ph = 1:nph
            @inbounds acc[ph, src.cell] -= phase_source(src, rhoS[ph], kr, mu, ph)
        end
    end
end

function insert_phase_sources!(acc::CuArray, kr, mu, rhoS, sources)
    nph = size(acc, 1)
    sources::CuArray
    # rhoS = CuArray(rhoS)
    i = map(cell, sources)
    for ph in 1:nph
        ρ = rhoS[ph]
        qi = map((src) -> phase_source(src, ρ, kr, mu, ph), sources)
        @. acc[ph, i] -= qi
    end
    CUDA.synchronize()
end

function convergence_criterion(model::SimulationModel{D, S}, storage, eq::ConservationLaw, r; dt = 1) where {D, S<:MultiPhaseSystem}
    n = number_of_equations_per_unit(eq)
    Φ = get_pore_volume(model)
    e = zeros(n)
    ρ = storage.state.PhaseMassDensities
    @tullio max e[j] := abs(r[j, i]) * dt / (value(ρ[j, i])*Φ[i])
    return (e, tolerance_scale(eq))
end

function get_reference_densities(model, storage)
    prm = storage.parameters
    if haskey(prm, :reference_densities)
        rhos = prm.reference_densities
    else
        rhos = ones(number_of_phases(model.system))
    end
    return rhos::AbstractVector{float_type(model.context)}
end
