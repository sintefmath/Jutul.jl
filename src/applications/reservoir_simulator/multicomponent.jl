using MultiComponentFlash
export TwoPhaseCompositionalSystem

get_components(sys::MultiComponentSystem) = sys.components
number_of_components(sys::MultiComponentSystem) = length(get_components(sys))

function degrees_of_freedom_per_entity(model::SimulationModel{G, S}, v::TotalMasses) where {G<:Any, S<:MultiComponentSystem}
    number_of_components(model.system)
end

function select_primary_variables_system!(S, domain, system::CompositionalSystem, formulation)
    S[:Pressure] = Pressure()
    S[:OverallMoleFractions] = OverallMoleFractions()
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


function apply_forces_to_equation!(storage, model::SimulationModel{D, S}, eq::ConservationLaw, force::V, time) where {V <: AbstractVector{SourceTerm{I, F, T}}, D, S<:TwoPhaseCompositionalSystem} where {I, F, T}
    acc = get_diagonal_entries(eq)
    state = storage.state
    kr = state.RelativePermeabilities
    rho = state.PhaseMassDensities
    mu = state.PhaseViscosities
    X = state.LiquidMassFractions
    Y = state.VaporMassFractions
    FR = state.FlashResults
    rhoS = get_reference_densities(model, storage)
    insert_component_sources!(acc, kr, mu, FR, X, Y, rho, rhoS, force)
end

function insert_component_sources!(acc, kr, mu, F, X, Y, rho, rhoS, sources)
    ncomp = size(acc, 1)
    for src in sources
        for c = 1:ncomp
            @inbounds acc[c, src.cell] -= component_source(src, kr, mu, F, X, Y, rho, rhoS, c)
        end
    end
end

function component_source(src, kr, mu, F, X, Y, rho, rhoS, c)
    # Treat inflow as volumetric with respect to surface conditions
    # Treat outflow as volumetric sources
    v = src.value
    cell = src.cell
    if v > 0
        f = src.fractional_flow[c]
    else
        f = compositional_out_f(kr, mu, X, Y, rho, c, cell, F[cell].state)
    end
    q = v*f
    @debug value(q)
    return q
end

function compositional_out_f(kr, mu, X, Y, rho, c, cell, ::TwoPhaseLiquidVapor)
    λ_l = local_mobility(kr, mu, 1, cell)
    λ_v = local_mobility(kr, mu, 2, cell)
    λ_t = λ_l + λ_v

    ρ_l = rho[1, cell]
    ρ_v = rho[2, cell]

    x = X[c, cell]
    y = Y[c, cell]

    return (λ_l*x*ρ_l + λ_v*y*ρ_v)/λ_t
end

function compositional_out_f(kr, mu, X, Y, rho, c, cell, ::SinglePhaseLiquid)
    ρ_l = rho[1, cell]
    y = X[c, cell]
    return y*ρ_l
end

function compositional_out_f(kr, mu, X, Y, rho, c, cell, ::SinglePhaseVapor)
    ρ_v = rho[2, cell]
    y = Y[c, cell]
    return y*ρ_v
end