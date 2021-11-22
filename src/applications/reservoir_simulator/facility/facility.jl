export TotalSurfaceMassRate, WellGroup, DisabledControl
export HistoryMode, PredictionMode, Wells



"""
Well variables - entities that we have exactly one of per well (and usually relates to the surface connection)
"""

function count_entities(wg::WellGroup, ::Wells)
    return length(wg.well_symbols)
end

function count_entities(::WellGroup, ::Any)
    error("Unit not found in well group.")
end

function get_domain_intersection(u::Cells, target_d::DiscretizedDomain{W}, source_d::WellControllerDomain,
                                           target_symbol, source_symbol) where {W<:WellGrid}
    # From controller to top well cell
    pos = get_well_position(source_d, target_symbol)
    if isnothing(pos)
        t = nothing
        s = nothing
    else
        t = [1]
        s = [pos]
    end
    (target = t, source = s, target_entity = u, source_entity = Wells())
end

function get_domain_intersection(u::Wells, target_d::WellControllerDomain, source_d::DiscretizedDomain{W},
                                           target_symbol, source_symbol) where {W<:WellGrid}
    # From top cell in well to control equation
    pos = get_well_position(target_d, source_symbol)
    if isnothing(pos)
        t = nothing
        s = nothing
    else
        t = [pos]
        s = [1]
    end
    (target = t, source = s, target_entity = u, source_entity = Cells())
end

function get_well_position(d, symbol)
    match = findall(d.well_symbols .== symbol)
    if length(match) == 0
        return nothing
    else
        return only(match)
    end
end

function associated_entity(::TotalSurfaceMassRate) Wells() end

function update_primary_variable!(state, massrate::TotalSurfaceMassRate, state_symbol, model, dx)
    v = state[state_symbol]
    symbols = model.domain.well_symbols
    cfg = state.WellGroupConfiguration.control
    # Injectors can only have strictly positive injection rates,
    # producers can only have strictly negative and disabled controls give zero rate.
    function do_update(v, dx, ctrl)
        return update_value(v, dx)
    end
    function do_update(v, dx, ctrl::InjectorControl)
        return update_value(v, dx, nothing, nothing, 1e-20, nothing)
    end
    function do_update(v, dx, ctrl::ProducerControl)
        return update_value(v, dx, nothing, nothing, nothing, -1e-20)
    end
    function do_update(v, dx, ctrl::DisabledControl)
        return 0.0
    end
    @inbounds for i in eachindex(v)
        s = symbols[i]
        v[i] = do_update(v[i], dx[i], cfg[s])
    end
end


function well_control_equation(ctrl, t::BottomHolePressureTarget, qt)
    # Note: This equation will get the bhp subtracted when coupled to a well
    return t.value
end

function well_control_equation(ctrl, t::SinglePhaseRateTarget, qt)
    # Note: This equation will get the corresponding phase rate subtracted by the well
    return t.value
end


function well_control_equation(ctrl, t::DisabledTarget, qt)
    return qt
end

## Well controls

"""
Impact from well group in facility on conservation equation inside well
"""
function update_cross_term!(ct::InjectiveCrossTerm, eq::ConservationLaw, well_storage, facility_storage,
                            target_model::SimulationModel{D, S}, source_model::SimulationModel{WG},
                            well_symbol, source, dt) where
                            {D<:DiscretizedDomain{W} where W<:WellGrid,
                            S<:Union{ImmiscibleSystem, SinglePhaseSystem},
                            WG<:WellGroup}
    fstate = facility_storage.state
    wstate = well_storage.state
    # Stuff from facility
    mswell = source_model.domain
    pos = get_well_position(mswell, well_symbol)
    ctrl = fstate.WellGroupConfiguration.control[well_symbol]
    qT = fstate.TotalSurfaceMassRate[pos]

    if isa(ctrl, InjectorControl)
        if value(qT) < 0
            @warn "Injector $well_symbol is producing?"
        end
        mix = ctrl.injection_mixture
        @assert length(mix) == number_of_phases(target_model.system) "Injection composition must match number of phases."
    else
        if value(qT) > 0
            @warn "Producer $well_symbol is injecting?"
        end
        top_node = 1
        masses = wstate.TotalMasses[:, top_node]
        mass = sum(masses)
        mix = masses./mass
    end
    update_topnode_sources!(ct.crossterm_source, ct.crossterm_target, qT, mix)
end

function update_topnode_sources!(cts, ctt, qT, mix)
    @inbounds for i in eachindex(mix)
        m = -mix[i]
        cts[i] = value(m)*qT
        ctt[i] = m*value(qT)
    end
end

"""
Cross term from well on control equation for well
"""
function update_cross_term!(ct::InjectiveCrossTerm, eq::ControlEquationWell,
                            target_storage, source_storage,
                            target_model::SimulationModel{WG},
                            source_model::SimulationModel{D},
                            target, well_symbol, dt) where {D<:DiscretizedDomain{W} where W<:WellGrid, WG<:WellGroup}
    fstate = target_storage.state
    ctrl = fstate.WellGroupConfiguration.control[well_symbol]
    target = ctrl.target
    # q_t = facility_surface_mass_rate_for_well(target_model, well_symbol, fstate)
    well_state = source_storage.state
    param = source_storage.parameters
    rhoS = param[:reference_densities]

    update_facility_control_crossterm!(ct.crossterm_source, ct.crossterm_target, well_state, rhoS, target_model, source_model, target, well_symbol, fstate)
    # # Operation twice, to get both partials
    # ct.crossterm_source[1] = -well_target(target, source_model, rhoS, value(q_t), well_state, x -> x)
    # ct.crossterm_target[1] = -well_target(target, source_model, rhoS, q_t, well_state, value)
end

function update_facility_control_crossterm!(s_buf, t_buf, well_state, rhoS, target_model, source_model, target, well_symbol, fstate)
    q_t = facility_surface_mass_rate_for_well(target_model, well_symbol, fstate)

    # well_state = source_storage.state
    # param = source_storage.parameters
    # rhoS = param[:reference_densities]
    # Operation twice, to get both partials
    s_buf[1] = -well_target(target, source_model, rhoS, value(q_t), well_state, x -> x)
    t_buf[1] = -well_target(target, source_model, rhoS, q_t, well_state, value)

end

function facility_surface_mass_rate_for_well(model::SimulationModel, wsym, fstate)
    pos = get_well_position(model.domain, wsym)
    return fstate.TotalSurfaceMassRate[pos]
end

bottom_hole_pressure(ws) = ws.Pressure[1]

function top_node_component_mass_fraction(ws, c_ix)
    tm = ws.TotalMasses
    t = ws.TotalMass
    mass_fraction = tm[1, c_ix]/t[1]
    return mass_fraction
end

well_target(target, well_model, rhoS, q_t, well_state, f) = 0.0
well_target(target::BottomHolePressureTarget, well_model, rhoS, q_t, well_state, f) = f(bottom_hole_pressure(well_state))

function well_target(target::SinglePhaseRateTarget, well_model::SimulationModel{D, S}, rhoS, q_t, well_state, f) where {D, S<:Union{ImmiscibleSystem, SinglePhaseSystem}}
    phases = get_phases(well_model.system)
    pos = findfirst(isequal(target.phase), phases)
    @assert !isnothing(pos)

    if value(q_t) >= 0
        q = q_t
    else
        mf = f(top_node_component_mass_fraction(well_state, pos))
        q = q_t*mf
    end
    return q/rhoS[pos]
end

function associated_entity(::ControlEquationWell) Wells() end

function update_equation!(eq::ControlEquationWell, storage, model, dt)
    state = storage.state
    ctrl = state.WellGroupConfiguration.control
    wells = model.domain.well_symbols
    # T = ctrl.target
    surf_rate = state.TotalSurfaceMassRate
    # bhp = state.Pressure[1]
    @inbounds for (i, key) in enumerate(wells)
        C = ctrl[key]
        T = C.target
        # @debug "Well $key operating using $T"
        eq.equation.entries[i] = well_control_equation(ctrl, T, surf_rate[i])
    end
end

function align_to_jacobian!(eq::ControlEquationWell, jac, model, u::Cells; kwarg...)
    # Need to align to cells, faces is automatically done since it is on the diagonal bands
    cache = eq.equation_top_cell
    layout = matrix_layout(model.context)
    control_equation_top_cell_alignment!(cache, jac, layout; kwarg...)
end

function control_equation_top_cell_alignment!(cache, jac, layout; equation_offset = 0, variable_offset = 0)
    nu, ne, np = ad_dims(cache)
    cellix = 1
    for e in 1:ne
        for d = 1:np
            pos = find_jac_position(jac, 1 + equation_offset, cellix + variable_offset, e, d, nu, nu, ne, np, layout)
            set_jacobian_pos!(cache, 1, e, d, pos)
        end
    end
end

# Selection of primary variables
function select_primary_variables_domain!(S, domain::WellGroup, system, formulation)
    S[:TotalSurfaceMassRate] = TotalSurfaceMassRate()
end

function select_equations_domain!(eqs, domain::WellGroup, system, arg...)
    # eqs[:potential_balance] = (PotentialDropBalanceWell, 1)
    eqs[:control_equation] = (ControlEquationWell, 1)
end

function build_forces(model::SimulationModel{D}; control = Dict(), limits = Dict()) where {D <: WellGroup}
    return (control = control::Dict, limits = limits::Dict,)
end

function initialize_extra_state_fields_domain!(state, model, domain::WellGroup)
    # Insert structure that holds well control (limits etc) that is then updated before each step
    state[:WellGroupConfiguration] = WellGroupConfiguration(domain.well_symbols)
end

function update_before_step_domain!(storage, model::SimulationModel, domain::WellGroup, dt, forces)
    # Set control to whatever is on the forces
    cfg = storage.state.WellGroupConfiguration
    for key in keys(forces.control)
        cfg.control[key] = forces.control[key]
    end
    for key in keys(forces.limits)
        cfg.limits[key] = forces.limits[key]
    end
end
