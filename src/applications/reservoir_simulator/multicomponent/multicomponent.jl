using MultiComponentFlash
export MultiPhaseCompositionalSystemLV
export StandardVolumeSource, VolumeSource, MassSource

const MINIMUM_COMPOSITIONAL_SATURATION = 1e-10

include("variables/variables.jl")
include("utils.jl")
include("flux.jl")
include("sources.jl")

function select_primary_variables_system!(S, domain, system::CompositionalSystem, formulation)
    S[:Pressure] = Pressure(max_rel = 0.2, minimum = 101325)
    S[:OverallMoleFractions] = OverallMoleFractions(dz_max = 0.1)
    if has_other_phase(system)
        S[:ImmiscibleSaturation] = ImmiscibleSaturation(ds_max = 0.2)
    end
end

function select_equations_system!(eqs, domain, system::MultiComponentSystem, formulation)
    nc = number_of_components(system)
    eqs[:mass_conservation] = (ConservationLaw, nc)
end

function select_secondary_variables_system!(S, domain, system::CompositionalSystem, formulation)
    select_default_darcy!(S, domain, system, formulation)
    if has_other_phase(system)
        water_pvt = ConstMuBTable(101325.0, 1.0, 1e-18, 1e-3, 1e-20)
        S[:PhaseMassDensities] = ThreePhaseCompositionalDensitiesLV(water_pvt)
        S[:PhaseViscosities] = ThreePhaseLBCViscositiesLV(water_pvt)
    else
        S[:PhaseMassDensities] = TwoPhaseCompositionalDensities()
        S[:PhaseViscosities] = LBCViscosities()
    end
    S[:LiquidMassFractions] = PhaseMassFractions(:liquid)
    S[:VaporMassFractions] = PhaseMassFractions(:vapor)
    S[:FlashResults] = FlashResults(system)
    S[:Saturations] = Saturations()
    S[:Temperature] = ConstantVariables(273.15 + 30.0)
end

function convergence_criterion(model::SimulationModel{D, S}, storage, eq::ConservationLaw, r; dt = 1) where {D, S<:CompositionalSystem}
    tm = storage.state0.TotalMasses
    a = active_entities(model.domain, Cells())
    function scale(i)
        @inbounds c = a[i]
        t = 0.0
        @inbounds for i = 1:size(tm, 1)
            t += tm[i, c]
        end
        return t
    end
    @tullio max e[j] := abs(r[j, i]) * dt / scale(i)
    names = model.system.equation_of_state.mixture.component_names
    R = Dict("CNV" => (errors = e, names = names))
    return R
end

