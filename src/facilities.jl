export TotalSurfaceMassRate, WellGroup
export HistoryMode, PredictionMode, Wells

abstract type FacilitySystem <: TervSystem end
struct PredictionMode <: FacilitySystem end
struct HistoryMode <: FacilitySystem end

abstract type SurfaceFacilityDomain <: TervDomain end
abstract type WellControllerDomain <: SurfaceFacilityDomain end
struct WellGroup <: WellControllerDomain
    well_symbols::Vector{Symbol}
end

"""
Well variables - units that we have exactly one of per well (and usually relates to the surface connection)
"""
struct Wells <: TervUnit end

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
    (target = 1, source = pos, source_unit = Wells())
end

function get_domain_intersection(u::Wells, target_d::WellControllerDomain, source_d::DiscretizedDomain{W},
                                           target_symbol, source_symbol) where {W<:WellGrid}
    # From top cell in well to control equation
    pos = get_well_position(target_d, source_symbol)
    (target = pos, source = 1, source_unit = Cells())
end

function get_well_position(d, symbol)
    return findall(d.well_symbols .== symbol)[]
end


# Bottom hole pressure for the well
# struct BottomHolePressure <: ScalarVariable end
# function associated_unit(::BottomHolePressure) Well() end

# Phase rates for well at surface conditions
# struct SurfacePhaseRates <: GroupedVariables end
# function associated_unit(::SurfacePhaseRates) Well() end

struct TotalSurfaceMassRate <: ScalarVariable end
function associated_unit(::TotalSurfaceMassRate) Wells() end

# function degrees_of_freedom_per_unit(model, v::SurfacePhaseRates)
#    return number_of_phases(model.system)
# end


abstract type WellTarget end

struct BottomHolePressureTarget <: WellTarget
    value::AbstractFloat
end

function well_control_equation(ctrl, t::BottomHolePressureTarget, qt)
    # Note: This equation will get the bhp subtracted when coupled to a well
    return t.value
end

struct SinglePhaseRateTarget <: WellTarget
    value::AbstractFloat
    phase::AbstractPhase
end

function well_control_equation(ctrl, t::SinglePhaseRateTarget, qt)
    # TODO: Add some notion of surface density
    return t.value - qt
end

## Well controls
abstract type WellForce <: TervForce end
abstract type WellControlForce <: WellForce end

struct DisabledControl <: WellControlForce

end

struct InjectorControl <: WellControlForce
    target::WellTarget
    injection_mixture
end

struct ProducerControl <: WellControlForce
    target::WellTarget
end

struct WellGroupConfiguration
    control
    limits
    function WellGroupConfiguration(well_symbols, control = nothing, limits = nothing)
        if isnothing(control)
            control = Dict{Symbol, WellControlForce}()
            for s in well_symbols
                control[s] = DisabledControl()
            end
        end
        if isnothing(limits)
            limits = Dict{Symbol, Any}()
            for s in well_symbols
                limits[s] = nothing
            end
        end
        new(control, limits)
    end
end

struct ControlEquationWell <: TervEquation
    # Equation:
    #        q_t - target = 0
    #        p|top cell - target = 0
    # We need to store derivatives with respect to q_t (same unit) and the top cell (other unit)
    equation::TervAutoDiffCache
    function ControlEquationWell(model, number_of_equations; kwarg...)
        # @assert number_of_equations == 1
        nw = count_units(model.domain, Wells())
        alloc = (unit) -> CompactAutoDiffCache(number_of_equations, nw, model, unit = unit; kwarg...)
        # One potential drop per velocity
        target_well = alloc(Wells())
        new(target_well)
    end
end

function update_cross_term!(ct::InjectiveCrossTerm, eq::ControlEquationWell, 
                            target_storage, source_storage,
                            target_model::SimulationModel{WG},
                            source_model::SimulationModel{D}, 
                            target, source, dt) where {D<:DiscretizedDomain{W} where W<:MultiSegmentWell, WG<:WellGroup}
    error("To be implemented - control eq.")
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
        @debug "Well $key operating using $T"
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
