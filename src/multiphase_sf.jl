export PhaseMassDensities, PhaseMobilities, MassMobilities

abstract type PhaseStateFunction <: TervStateFunction end
abstract type ComponentStateFunction <: TervStateFunction end
abstract type PhaseAndComponentStateFunction <: TervStateFunction end

function degrees_of_freedom_per_unit(model, sf::PhaseStateFunction) number_of_phases(model.system) end

# Single-phase specialization
function degrees_of_freedom_per_unit(model::SimulationModel{D, S}, sf::ComponentStateFunction) where {D, S<:SinglePhaseSystem} 1 end
function degrees_of_freedom_per_unit(model::SimulationModel{D, S}, sf::PhaseAndComponentStateFunction) where {D, S<:SinglePhaseSystem} 1 end

# Immiscible specialization
function degrees_of_freedom_per_unit(model::SimulationModel{D, S}, sf::ComponentStateFunction) where {D, S<:ImmiscibleSystem}
    number_of_phases(model.system)
end
function degrees_of_freedom_per_unit(model::SimulationModel{D, S}, sf::PhaseAndComponentStateFunction) where {D, S<:ImmiscibleSystem}
    number_of_phases(model.system)
end

function select_state_functions!(sf, system::MultiPhaseSystem)
    sf[:PhaseMassDensities] = PhaseMassDensities()
    sf[:PhaseMobilities] = PhaseMobilities()
    sf[:MassMobilities] = MassMobilities()
    sf[:TotalMasses] = TotalMasses()
end

"""
Volumetric mobility of each phase
"""
struct PhaseMobilities <: PhaseStateFunction end

function update_self!(mob, tv::PhaseMobilities, model::SimulationModel{G, S}, state, param) where {G, S<:SinglePhaseSystem}
    mu = param.Viscosity[1]
    fapply!(mob, () -> 1/mu)
end

function update_self!(mob, tv::PhaseMobilities, model::SimulationModel{G, S}, state, param) where {G, S<:ImmiscibleSystem}
    n = param.CoreyExponents
    mu = param.Viscosity
    s = state.Saturations
    @. mob = s^n/mu
end

"""
Mass density of each phase
"""
struct PhaseMassDensities <: PhaseStateFunction end

function update_self!(rho, tv::PhaseMassDensities, model, state, param)
    rho_input = param.Density
    p = state.Pressure
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
end

"""
Mobility of the mass of each component, in each phase (TBD how to represent this in general)
"""
struct MassMobilities <: PhaseAndComponentStateFunction end

function update_self!(mobrho, tv::MassMobilities, model, state, param)
    mobrho = state.MassMobilities
    mob = state.PhaseMobilities
    rho = state.PhaseMassDensities
    fapply!(mobrho, *, mob, rho)
end

"""
Total mass of each component/species/... 
"""
function update_self!(totmass, tv::TotalMasses, model::SimulationModel{G, S}, state, param) where {G, S<:SinglePhaseSystem}
    pv = get_pore_volume(model)
    rho = state.PhaseMassDensities
    fapply!(totmass, *, rho, pv)
end

function update_self!(totmass, tv::TotalMasses, model::SimulationModel{G, S}, state, param) where {G, S<:ImmiscibleSystem}
    pv = get_pore_volume(model)
    rho = state.PhaseMassDensities
    s = state.Saturations
    @. totmass = rho*pv*s
end
