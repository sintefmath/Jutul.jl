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
function update_mobility!(storage, model::SimulationModel{G, S}) where {G, S<:SinglePhaseSystem}
    mob = storage.properties.Mobility
    mu = storage.parameters.Viscosity[1]
    fapply!(mob, () -> 1/mu)
end

function update_mobility!(storage, model::SimulationModel{G, S}) where {G, S<:ImmiscibleSystem}
    p = storage.parameters
    mob = storage.properties.Mobility
    n = p.CoreyExponents
    mu = p.Viscosity

    s = storage.state.Saturations
    @. mob = s^n/mu
end
"""
Mass density of each phase
"""
struct PhaseMassDensities <: PhaseStateFunction end

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

"""
Mobility of the mass of each component, in each phase (TBD how to represent this in general)
"""
struct MassMobilities <: PhaseAndComponentStateFunction end
function update_mass_mobility!(storage, model)
    props = storage.properties
    mobrho = props.MassMobility
    mob = props.Mobility
    rho = props.Density
    # Assign the values
    fapply!(mobrho, *, mob, rho)
end

"""
Total mass of each component/species/... 
"""
# struct TotalMasses <: ComponentStateFunction end
function update_total_masses!(storage, model::SimulationModel{G, S}) where {G<:Any, S<:SinglePhaseSystem}
    pv = get_pore_volume(model)
    rho = storage.properties.Density
    totMass = storage.state.TotalMasses
    fapply!(totMass, *, rho, pv)
end

function update_total_masses!(storage, model::SimulationModel{G, S}) where {G<:Any, S<:ImmiscibleSystem}
    state = storage.state
    pv = get_pore_volume(model)
    rho = storage.properties.Density
    totMass = state.TotalMasses
    s = state.Saturations
    @. totMass = rho*pv*s
end
