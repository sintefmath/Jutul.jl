# Saturations as primary variable
struct OverallMoleFractions <: FractionVariables
    dz_max
    OverallMoleFractions(;dz_max = 0.2) = new(dz_max)
end

values_per_entity(model, v::OverallMoleFractions) = number_of_components(model.system)

minimum_value(::OverallMoleFractions) = MultiComponentFlash.MINIMUM_COMPOSITION
absolute_increment_limit(z::OverallMoleFractions) = z.dz_max


function update_primary_variable!(state, p::OverallMoleFractions, state_symbol, model, dx)
    s = state[state_symbol]
    unit_sum_update!(s, p, model, dx)
end

mutable struct InPlaceFlashBuffer
    z
    forces
    function InPlaceFlashBuffer(n)
        z = zeros(n)
        new(z, nothing)
    end
end

struct FlashResults <: ScalarVariable
    storage
    method
    update_buffer
    function FlashResults(system; method = SSIFlash(), kwarg...)
        eos = system.equation_of_state
        # np = number_of_partials_per_entity(system, Cells())
        n = MultiComponentFlash.number_of_components(eos)
        storage = []
        buffers = []
        N = Threads.nthreads()
        for i = 1:N
            s = flash_storage(eos, method = method, inc_jac = true, diff_externals = true, npartials = n, static_size = true)
            push!(storage, s)
            b = InPlaceFlashBuffer(n)
            push!(buffers, b)
        end
        storage = tuple(storage...)
        buffers = tuple(buffers...)
        new(storage, method, buffers)
    end
end

default_value(model, ::FlashResults) = FlashedMixture2Phase(model.system.equation_of_state)

function initialize_variable_value!(state, model, pvar::FlashResults, symb, val::AbstractDict; need_value = false)
    @assert need_value == false
    n = number_of_entities(model, pvar)
    v = default_value(model, pvar)
    T = typeof(v)
    V = Vector{T}(undef, n)
    for i in 1:n
        V[i] = default_value(model, pvar)
    end
    # V = repeat([v], n)
    initialize_variable_value!(state, model, pvar, symb, V)
end

function initialize_variable_ad(state, model, pvar::FlashResults, symb, npartials, diag_pos; context = DefaultContext(), kwarg...)
    n = number_of_entities(model, pvar)
    v_ad = get_ad_entity_scalar(1.0, npartials, diag_pos; kwarg...)
    ∂T = typeof(v_ad)
    eos = model.system.equation_of_state

    r = FlashedMixture2Phase(eos, ∂T)
    T = typeof(r)
    # T = MultiComponentFlash.flashed_mixture_array_type(eos, ∂T)
    V = Vector{T}(undef, n)
    for i in 1:n
        V[i] = FlashedMixture2Phase(eos, ∂T)
    end
    state[symb] = V
    return state
end

struct TwoPhaseCompositionalDensities <: PhaseMassDensities
end

struct PhaseMassFractions{T} <: FractionVariables
    phase::T
end

struct LBCViscosities <: PhaseVariables end

values_per_entity(model, v::PhaseMassFractions) = number_of_components(model.system)

function select_secondary_variables_system!(S, domain, system::CompositionalSystem, formulation)
    select_default_darcy!(S, domain, system, formulation)
    S[:PhaseMassDensities] = TwoPhaseCompositionalDensities()
    S[:LiquidMassFractions] = PhaseMassFractions(:liquid)
    S[:VaporMassFractions] = PhaseMassFractions(:vapor)
    S[:FlashResults] = FlashResults(system)
    S[:Saturations] = Saturations()
    S[:Temperature] = ConstantVariables([273.15 + 30.0])
    S[:PhaseViscosities] = LBCViscosities()
end

@terv_secondary function update_as_secondary!(flash_results, fr::FlashResults, model, param, Pressure, Temperature, OverallMoleFractions)
    storage, m, buffers = fr.storage, fr.method, fr.update_buffer
    eos = model.system.equation_of_state

    for buf in buffers
        update_flash_buffer!(buf, eos, Pressure, Temperature, OverallMoleFractions)
    end
    perform_flash_for_all_cells!(flash_results, storage, m, eos, buffers, Pressure, Temperature, OverallMoleFractions)
end

function perform_flash_for_all_cells!(flash_results, storage, m, eos, buffers, P, T, z; threads = true)
    flash_cell(i, S, buf) = internal_flash!(flash_results, S, m, eos, buf, P, T, z, i)
    if threads
        @inbounds @batch threadlocal=thread_buffers(storage, buffers) for i in eachindex(flash_results)
            # Unpack thread specific storage
            S, thread_buffer = threadlocal
            # Do flash
            flash_cell(i, S, thread_buffer)
        end
    else
        S, buf = thread_buffers(storage, buffers)
        @inbounds for i in eachindex(flash_results)
            flash_cell(i, S, buf)
        end
    end
end

function thread_buffers(storage, buffers)
    thread_id = Threads.threadid()
    S = storage[thread_id]
    thread_buffer = buffers[thread_id]
    return (S, thread_buffer)
end

function update_flash_buffer!(buf, eos, Pressure, Temperature, OverallMoleFractions)
    if isnothing(buf.forces) || eltype(buf.forces.A_ij) != eltype(OverallMoleFractions)
        P = Pressure[1]
        T = Temperature[1]
        Z = OverallMoleFractions[:, 1]
        buf.forces = force_coefficients(eos, (p = P, T = T, z = Z), static_size = true)
    end
end

function internal_flash!(flash_results, S, m, eos, buf, Pressure, Temperature, OverallMoleFractions, i)
    # Ready to work
    @inbounds begin
        f = flash_results[i]
        P = Pressure[i]
        T = Temperature[i]
        Z = @view OverallMoleFractions[:, i]

        K = f.K
        x = f.liquid.mole_fractions
        y = f.vapor.mole_fractions

        flash_results[i] = update_flash_result(S, m, buf, eos, K, x, y, buf.z, buf.forces, P, T, Z)
    end
end


function update_flash_result(S, m, buffer, eos, K, x, y, z, forces, P, T, Z)
    @. z = value(Z)
    # Conditions
    c = (p = value(P), T = value(T), z = z)
    # Perform flash
    vapor_frac = flash_2ph!(S, K, eos, c, NaN, method = m, extra_out = false)
    force_coefficients!(forces, eos, (p = P, T = T, z = Z))
    if isnan(vapor_frac)
        # Single phase condition. Life is easy.
        Z_L, Z_V, V, phase_state = single_phase_update!(P, T, Z, x, y, forces, eos, c)
    else
        # Two-phase condition: We have some work to do.
        Z_L, Z_V, V, phase_state = two_phase_update!(S, P, T, Z, x, y, K, vapor_frac, forces, eos, c)
    end
    out = FlashedMixture2Phase(phase_state, K, V, x, y, Z_L, Z_V)
    return out
end

function get_compressibility_factor(forces, eos, P, T, Z)
    ∂cond = (p = P, T = T, z = Z)
    force_coefficients!(forces, eos, ∂cond)
    return mixture_compressibility_factor(eos, ∂cond, forces)
end

@inline function single_phase_update!(P, T, Z, x, y, forces, eos, c)
    AD = Base.promote_type(eltype(Z), typeof(P), typeof(T))
    Z_L = get_compressibility_factor(forces, eos, P, T, Z)
    Z_V = Z_L
    @. x = Z
    @. y = Z
    V = single_phase_label(eos.mixture, c)
    if V > 0.5
        phase_state = MultiComponentFlash.single_phase_v
    else
        phase_state = MultiComponentFlash.single_phase_l
    end
    V = convert(AD, V)
    out = (Z_L::AD, Z_V::AD, V::AD, phase_state::PhaseState2Phase)
    return out
end

function two_phase_pre!(S, P, T, Z, x::AbstractVector{AD}, y::AbstractVector{AD}, vapor_frac, eos, c) where {AD <: ForwardDiff.Dual}
    inverse_flash_update!(S, eos, c, vapor_frac)
    ∂c = (p = P, T = T, z = Z)
    V = set_partials_vapor_fraction(convert(AD, vapor_frac), S, eos, ∂c)
    set_partials_phase_mole_fractions!(x, S, eos, ∂c, :liquid)
    set_partials_phase_mole_fractions!(y, S, eos, ∂c, :vapor)
    return V
end

two_phase_pre!(S, P, T, Z, x, y, V, eos, c) = V


@inline function two_phase_update!(S, P, T, Z, x, y, K, vapor_frac, forces, eos, c)
    AD = Base.promote_type(typeof(P), eltype(Z), typeof(T))
    @. x = liquid_mole_fraction(Z, K, vapor_frac)
    @. y = vapor_mole_fraction(x, K)
    V = two_phase_pre!(S, P, T, Z, x, y, vapor_frac, eos, c)
    Z_L = get_compressibility_factor(forces, eos, P, T, x)
    Z_V = get_compressibility_factor(forces, eos, P, T, y)
    phase_state = MultiComponentFlash.two_phase_lv

    return (Z_L::AD, Z_V::AD, V::AD, phase_state)
end

@terv_secondary function update_as_secondary!(Sat, s::Saturations, model::SimulationModel{D, S}, param, FlashResults) where {D, S<:CompositionalSystem}
    tb = thread_batch(model.context)
    @inbounds @batch minbatch = tb for i in 1:size(Sat, 2)
        S_l, S_v = phase_saturations(FlashResults[i])
        Sat[1, i] = S_l
        Sat[2, i] = S_v
    end
end

@terv_secondary function update_as_secondary!(X, m::PhaseMassFractions, model::SimulationModel{D, S}, param, FlashResults) where {D, S<:CompositionalSystem}
    molar_mass = map((x) -> x.mw, model.system.equation_of_state.mixture.properties)
    phase = m.phase
    tb = thread_batch(model.context)
    @inbounds @batch minbatch = tb for i in eachindex(FlashResults)
        f = FlashResults[i]
        if phase_is_present(phase, f.state)
            X_i = view(X, :, i)
            r = phase_data(f, phase)
            x_i = r.mole_fractions
            update_mass_fractions!(X_i, x_i, molar_mass)
        end
    end
end

@inline function update_mass_fractions!(X, x, molar_masses)
    t = zero(eltype(X))
    @inbounds for i in eachindex(x)
        tmp = molar_masses[i]*x[i]
        t += tmp
        X[i] = tmp
    end
    @. X = X/t
end

@terv_secondary function update_as_secondary!(rho, m::TwoPhaseCompositionalDensities, model::SimulationModel{D, S}, param, Pressure, Temperature, FlashResults) where {D, S<:CompositionalSystem}
    eos = model.system.equation_of_state
    n = size(rho, 2)
    tb = thread_batch(model.context)
    @inbounds @batch minbatch = tb for i in 1:n
        p = Pressure[i]
        T = Temperature[1, i]
        rho[1, i], rho[2, i] = mass_densities(eos, p, T, FlashResults[i])
    end
end

@terv_secondary function update_as_secondary!(mu, m::LBCViscosities, model::SimulationModel{D, S}, param, Pressure, Temperature, FlashResults) where {D, S<:CompositionalSystem}
    eos = model.system.equation_of_state
    tb = thread_batch(model.context)
    @inbounds @batch minbatch = tb for i in 1:size(mu, 2)
        p = Pressure[i]
        T = Temperature[1, i]
        mu[1, i], mu[2, i] = lbc_viscosities(eos, p, T, FlashResults[i])
    end
end

# Total masses
@terv_secondary function update_as_secondary!(totmass, tv::TotalMasses, model::SimulationModel{G, S}, param, 
                                                                                                    FlashResults,
                                                                                                    PhaseMassDensities,
                                                                                                    Saturations,
                                                                                                    VaporMassFractions,
                                                                                                    LiquidMassFractions,
                                                                                                    FluidVolume) where {G, S<:CompositionalSystem}
    pv = FluidVolume
    ρ = PhaseMassDensities
    X = LiquidMassFractions
    Y = VaporMassFractions
    Sat = Saturations
    F = FlashResults

    @tullio totmass[c, i] = two_phase_compositional_mass(F[i].state, ρ, X, Y, Sat, c, i)*pv[i]
end

function two_phase_compositional_mass(state, ρ, X, Y, S, c, i)
    T = eltype(ρ)
    if liquid_phase_present(state)
        @inbounds M_l = ρ[1, i]*S[1, i]*X[c, i]
    else
        M_l = zero(T)
    end

    if vapor_phase_present(state)
        @inbounds M_v = ρ[2, i]*S[2, i]*Y[c, i]
    else
        M_v = zero(T)
    end
    return M_l + M_v
end
