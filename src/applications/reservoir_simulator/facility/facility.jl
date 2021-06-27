export TotalSurfaceMassRate, WellGroup, DisabledControl
export HistoryMode, PredictionMode, Wells



"""
Well variables - units that we have exactly one of per well (and usually relates to the surface connection)
"""

function count_units(wg::WellGroup, ::Wells)
    return length(wg.well_symbols)
end

function count_units(::WellGroup, ::Any)
    error("Unit not found in well group.")
end

function get_domain_intersection(u::Cells, target_d::DiscretizedDomain{W}, source_d::WellControllerDomain, 
                                           target_symbol, source_symbol) where {W<:WellGrid}
    # From controller to top well cell
    pos = get_well_position(source_d, target_symbol)
    (target = 1, source = pos, target_unit = u, source_unit = Wells())
end

function get_domain_intersection(u::Wells, target_d::WellControllerDomain, source_d::DiscretizedDomain{W},
                                           target_symbol, source_symbol) where {W<:WellGrid}
    # From top cell in well to control equation
    pos = get_well_position(target_d, source_symbol)
    (target = pos, source = 1, target_unit = u, source_unit = Cells())
end

function get_well_position(d, symbol)
    return findall(d.well_symbols .== symbol)[]
end

function associated_unit(::TotalSurfaceMassRate) Wells() end

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
    for i in eachindex(v)
        s = symbols[i]
        v[i] = do_update(v[i], dx[i], cfg[s])
    end
end


function well_control_equation(ctrl, t::BottomHolePressureTarget, qt)
    # Note: This equation will get the bhp subtracted when coupled to a well
    return t.value
end

function well_control_equation(ctrl, t::SinglePhaseRateTarget, qt)
    # TODO: Add some notion of surface density
    return t.value - qt
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

    function update_topnode_sources!(src, qT, mix)
        for i in eachindex(mix)
            src[i] = -mix[i]*qT
        end
    end
    update_topnode_sources!(ct.crossterm_source, qT, value.(mix))
    update_topnode_sources!(ct.crossterm_target, value(qT), mix)
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
    if isa(target, BottomHolePressureTarget)
        # Treat top node as bhp reference point
        bhp_contribution = -source_storage.state.Pressure[1]

        # Cross-term in well
        ct.crossterm_source[1] = bhp_contribution
        ct.crossterm_target[1] = value(bhp_contribution)
    end
end

function associated_unit(::ControlEquationWell) Wells() end

function update_equation!(eq::ControlEquationWell, storage, model, dt)
    state = storage.state
    ctrl = state.WellGroupConfiguration.control
    wells = model.domain.well_symbols
    # T = ctrl.target
    surf_rate = state.TotalSurfaceMassRate
    # bhp = state.Pressure[1]
    for (i, key) in enumerate(wells)
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
    for (key, val) in forces.control
        cfg.control[key] = val
    end
    for (key, val) in forces.limits
        cfg.limits[key] = val
    end
end
