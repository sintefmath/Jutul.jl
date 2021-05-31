export PhaseMassDensities, PhaseMobilities, MassMobilities

abstract type PhaseVariable <: GroupedVariables end
abstract type ComponentVariable <: GroupedVariables end
abstract type PhaseAndComponentVariable <: GroupedVariables end

function degrees_of_freedom_per_unit(model, sf::PhaseVariable) number_of_phases(model.system) end

# Single-phase specialization
function degrees_of_freedom_per_unit(model::SimulationModel{D, S}, sf::ComponentVariable) where {D, S<:SinglePhaseSystem} 1 end
function degrees_of_freedom_per_unit(model::SimulationModel{D, S}, sf::PhaseAndComponentVariable) where {D, S<:SinglePhaseSystem} 1 end

# Immiscible specialization
function degrees_of_freedom_per_unit(model::SimulationModel{D, S}, sf::ComponentVariable) where {D, S<:ImmiscibleSystem}
    number_of_phases(model.system)
end
function degrees_of_freedom_per_unit(model::SimulationModel{D, S}, sf::PhaseAndComponentVariable) where {D, S<:ImmiscibleSystem}
    number_of_phases(model.system)
end

function select_secondary_variables!(S, system::MultiPhaseSystem)
    S[:PhaseMassDensities] = PhaseMassDensities()
    S[:TotalMasses] = TotalMasses()
end

function select_secondary_variables!(S, domain::DiscretizedDomain{G}, system::MultiPhaseSystem, arg...) where {G <: PorousMediumGrid}
    select_secondary_variables!(S, system)
    # For a porous medium, we also need the notion of mobilities.
    S[:PhaseMobilities] = PhaseMobilities()
    S[:MassMobilities] = MassMobilities()
end

function minimum_output_variables(system::MultiPhaseSystem, primary_variables)
    [:TotalMasses]
end

"""
Volumetric mobility of each phase
"""
struct PhaseMobilities <: PhaseVariable end

@terv_secondary function update_as_secondary!(mob, tv::PhaseMobilities, model::SimulationModel{G, S}, param) where {G, S<:SinglePhaseSystem}
    mu = param.Viscosity[1]
    fapply!(mob, () -> 1/mu)
end

@terv_secondary function update_as_secondary!(mob, tv::PhaseMobilities, model::SimulationModel{G, S}, param, Saturations) where {G, S<:ImmiscibleSystem}
    n = param.CoreyExponents
    mu = param.Viscosity
    @. mob = Saturations^n/mu
end

"""
Mass density of each phase
"""
struct PhaseMassDensities <: PhaseVariable end

@terv_secondary function update_as_secondary!(rho, tv::PhaseMassDensities, model, param, Pressure)
    rho_input = param.Density
    p = Pressure
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
struct MassMobilities <: PhaseAndComponentVariable end

@terv_secondary function update_as_secondary!(mobrho, tv::MassMobilities, model, param, PhaseMobilities, PhaseMassDensities)
    fapply!(mobrho, *, PhaseMobilities, PhaseMassDensities)
end

# Total masses
@terv_secondary function update_as_secondary!(totmass, tv::TotalMasses, model::SimulationModel{G, S}, param, PhaseMassDensities) where {G, S<:SinglePhaseSystem}
    pv = get_pore_volume(model)
    fapply!(totmass, *, PhaseMassDensities, pv)
end

@terv_secondary function update_as_secondary!(totmass, tv::TotalMasses, model::SimulationModel{G, S}, param, PhaseMassDensities, Saturations) where {G, S<:ImmiscibleSystem}
    pv = get_pore_volume(model)
    rho = PhaseMassDensities
    s = Saturations
    @. totmass = rho*pv*s
end

# Total mass
@terv_secondary function update_as_secondary!(totmass, tv::TotalMass, model::SimulationModel{G, S}, param, TotalMasses) where {G, S<:MultiPhaseSystem}
    tmp = TotalMasses'
    sum!(totmass, tmp)
end
