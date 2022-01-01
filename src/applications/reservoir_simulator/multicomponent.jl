using MultiComponentFlash
export TwoPhaseCompositionalSystem
export StandardVolumeSource, VolumeSource, MassSource

get_components(sys::MultiComponentSystem) = sys.components
number_of_components(sys::MultiComponentSystem) = length(get_components(sys))

function degrees_of_freedom_per_entity(model::SimulationModel{G, S}, v::TotalMasses) where {G<:Any, S<:MultiComponentSystem}
    number_of_components(model.system)
end

function select_primary_variables_system!(S, domain, system::CompositionalSystem, formulation)
    S[:Pressure] = Pressure(max_rel = 0.2, minimum = 101325)
    S[:OverallMoleFractions] = OverallMoleFractions(dz_max = 0.1)
end

function select_equations_system!(eqs, domain, system::MultiComponentSystem, formulation)
    nc = number_of_components(system)
    eqs[:mass_conservation] = (ConservationLaw, nc)
end

function convergence_criterion(model::SimulationModel{D, S}, storage, eq::ConservationLaw, r; dt = 1) where {D, S<:TwoPhaseCompositionalSystem}
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
    Sat = state.Saturations
    FR = state.FlashResults
    rhoS = get_reference_densities(model, storage)
    insert_component_sources!(acc, Sat, kr, mu, FR, X, Y, rho, rhoS, force)
end

function insert_component_sources!(acc, S, kr, mu, F, X, Y, rho, rhoS, sources)
    ncomp = size(acc, 1)
    for src in sources
        for c = 1:ncomp
            @inbounds acc[c, src.cell] -= component_source(src, S, kr, mu, F, X, Y, rho, rhoS, c)
        end
    end
end

function component_source(src, S, kr, mu, F, X, Y, rho, rhoS, c)
    # Treat inflow as volumetric with respect to surface conditions
    # Treat outflow as volumetric sources
    v = src.value
    cell = src.cell
    # MassSource -> source terms are already masses
    # VolumeSource -> source terms are volumes at medium conditions
    t = src.type
    f_c = src.fractional_flow[c]
    if v > 0
        if t == MassSource
            # Source term is already as masses. Injection is straightforward, 
            # production is weighted according to mobility*density
            f = f_c
        elseif t == StandardVolumeSource
            f = rhoS[1]*f_c
        elseif t == VolumeSource
            # Saturation-averaged density
            ρ_mix = S[1, cell]*rho[1, cell] + S[2, cell]*rho[2, cell]
            f = f_c*ρ_mix
        else
            error("Not supported source term type: $t")
        end
    else
        f = compositional_out_f(kr, mu, X, Y, rho, rhoS, c, cell, t, F[cell].state)
    end
    q = v*f
    return q
end

function compositional_out_f(kr, mu, X, Y, rho, rhoS, c, cell, t, state)
    λ_l = local_mobility(kr, mu, 1, cell)
    λ_v = local_mobility(kr, mu, 2, cell)

    ρ_l = rho[1, cell]
    ρ_v = rho[2, cell]

    x = X[c, cell]
    y = Y[c, cell]

    if t == MassSource
        m_l = ρ_l*λ_l
        m_v = ρ_v*λ_v
        m_t = m_l + m_v
        f = (m_l*x + m_v*y)/m_t
    elseif t == VolumeSource
        λ_t = λ_l + λ_v
        f = (λ_l*x*ρ_l + λ_v*y*ρ_v)/λ_t
    elseif t == StandardVolumeSource
        ρ_ls = rhoS[1]
        ρ_vs = rhoS[2]
    
        λ_t = λ_l + λ_v
        f_l = λ_l/λ_t
        f_v = λ_v/λ_t
        f = ρ_ls*x*f_l + ρ_vs*y*f_v
    end
    return f
end

# function compositional_out_f(kr, mu, X, Y, rho, rhoS, c, cell, t, ::SinglePhaseLiquid)
#     ρ_l = rho[1, cell]
#     x = X[c, cell]
#     if t == MassSource
#         f = x
#     elseif t == VolumeSource
#         f = x*ρ_l
#     elseif t == StandardVolumeSource
#         f = rhoS[1]*x
#     end
#     return f
# end

# function compositional_out_f(kr, mu, X, Y, rho, rhoS, c, cell, t, ::SinglePhaseVapor)
#     ρ_v = rho[2, cell]
#     y = Y[c, cell]
#     if t == MassSource
#         f = y
#     elseif t == VolumeSource
#         f = y*ρ_v
#     elseif t == StandardVolumeSource
#         f = rhoS[2]*y
#     end
#     return f
# end
