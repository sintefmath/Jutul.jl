
function well_target(target::SurfaceVolumeTarget, well_model::SimulationModel{D, S}, rhoS, q_t, well_state, f) where {D, S<:MultiComponentSystem}
    phases = get_phases(well_model.system)
    positions = surface_target_phases(target, phases)

    if value(q_t) >= 0
        pos = only(positions)
        q = q_t/rhoS[pos]
    else
        @assert length(positions) > 0
        q = zero(q_t)
        rhoS_sep, S_sep = flash_well_densities(well_model, well_state, rhoS)
        for (v_frac, ρ, pos) in zip(S_sep, rhoS_sep, positions)
            q += q_t*f(v_frac/ρ)
        end
    end
    return q
end

function flash_well_densities(well_model, well_state, rhoS)
    total_masses = well_state.TotalMasses
    T = eltype(total_masses)
    nph = length(rhoS)
    rho = zeros(T, nph)
    volfrac = zeros(T, nph)
    sys = well_model.system
    eos = sys.equation_of_state
    nc = MultiComponentFlash.number_of_components(eos)

    if nph == 3
        a, l, v = phase_indices(sys)
        rho[a] = rhoS[a]
        S_other = well_state.Saturations[a, 1]
        volfrac[a] = S_other
        n = nc + 1
    else
        l, v = phase_indices(sys)
        n = nc
        S_other = zero(T)
    end
    buf = InPlaceFlashBuffer(nc)

    sc = well_model.domain.grid.surface
    Pressure = convert(T, sc.p)
    Temperature = convert(T, sc.T)

    mass = total_masses[1:nc, 1]
    mw = map(x -> x.mw, eos.mixture.properties)

    z = mass./mw
    zt = sum(z)
    @. z = z/zt
    m = SSIFlash()
    S = flash_storage(eos, method = m, inc_jac = true, diff_externals = true, npartials = n, static_size = true)
    update_flash_buffer!(buf, eos, Pressure, Temperature, z)

    f = FlashedMixture2Phase(eos, T)
    x = f.liquid.mole_fractions
    y = f.vapor.mole_fractions
    forces = buf.forces

    S = update_flash_result(S, m, buf, eos, f.K, x, y, buf.z, forces, Pressure, Temperature, z)

    rho[l], rho[v] = mass_densities(eos, Pressure, Temperature, S)
    rem = one(T) - S_other
    S_l, S_v = phase_saturations(S)

    volfrac[l] = rem*S_l
    volfrac[v] = rem*S_v
    return (rho, volfrac)
end