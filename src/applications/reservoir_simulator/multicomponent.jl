using MultiComponentFlash
export TwoPhaseCompositionalSystem

get_components(sys::MultiComponentSystem) = sys.components
number_of_components(sys::MultiComponentSystem) = length(get_components(sys))

function degrees_of_freedom_per_entity(model::SimulationModel{G, S}, v::TotalMasses) where {G<:Any, S<:MultiComponentSystem}
    number_of_components(model.system)
end

function select_primary_variables_system!(S, domain, system::CompositionalSystem, formulation)
    S[:Pressure] = Pressure()
    S[:OverallCompositions] = OverallCompositions()
end

function select_equations_system!(eqs, domain, system::MultiComponentSystem, formulation)
    nc = number_of_components(system)
    eqs[:mass_conservation] = (ConservationLaw, nc)
end

function convergence_criterion(model::SimulationModel{D, S}, storage, eq::ConservationLaw, r; dt = 1) where {D, S<:TwoPhaseCompositionalSystem}
    Φ = get_pore_volume(model)
    ρ = storage.state.PhaseMassDensities
    s = storage.state.Saturations
    @tullio max e[j] := abs(r[j, i]) * dt / (value(ρ[1, i]*s[1, i] + ρ[2, i]*s[2, i])*Φ[i])
    return (e, tolerance_scale(eq))
end

# function setup_storage_system!(storage, model, system::TwoPhaseCompositionalSystem)
#     s = model.system
#     eos = s.equation_of_state
#     n = MultiComponentFlash.number_of_components(eos.mixture)
#     c = (p = 101325, T = 273.15 + 20, z = zeros(n))
#     m = s.flash_method
#     np = number_of_partials_per_entity(model, Cells())
#     storage[:flash] = flash_storage(eos, c, m, inc_jac = true, diff_externals = true, npartials = np)
# end
