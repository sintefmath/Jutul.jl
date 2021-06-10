export TotalSurfaceMassRate


abstract type SurfaceFacilityDomain <: TervDomain end
abstract type WellControllerDomain <: SurfaceFacilityDomain end
struct SingleWellController <: WellControllerDomain end

"""
Well variables - units that we have exactly one of per well (and usually relates to the surface connection)
"""
struct Well <: TervUnit end

function count_units(::SingleWellController, ::Well)
    return 1
end

function count_units(::SingleWellController, ::Any)
    error("Unit not found in well domain.")
end




# Bottom hole pressure for the well
# struct BottomHolePressure <: ScalarVariable end
# function associated_unit(::BottomHolePressure) Well() end

# Phase rates for well at surface conditions
# struct SurfacePhaseRates <: GroupedVariables end
# function associated_unit(::SurfacePhaseRates) Well() end

struct TotalSurfaceMassRate <: ScalarVariable end
function associated_unit(::TotalSurfaceMassRate) Well() end

# function degrees_of_freedom_per_unit(model, v::SurfacePhaseRates)
#    return number_of_phases(model.system)
# end

function well_control_equation(t::BottomHolePressureTarget, qt, bhp)
    return bhp - t.value
end

struct SinglePhaseRateTarget <: WellTarget
    value::AbstractFloat
    phase::AbstractPhase
end

function well_control_equation(t::SinglePhaseRateTarget, qt, bhp)
    # Assuming injector
    return qt - t.value
end


## Well controls
abstract type WellForce <: TervForce end
abstract type WellControlForce <: WellForce end

struct InjectorControl <: WellControlForce
    target::WellTarget
    injection_mixture
end

struct ProducerControl <: WellControlForce
    target::WellTarget
end

mutable struct WellConfiguration
    control
    target
    limits
    function WellConfiguration(control = nothing, target = nothing, limits = nothing)
        new(control, target, limits)
    end
end

struct ControlEquationWell <: TervEquation
    # Equation:
    #        q_t - target = 0
    #        p|top cell - target = 0
    # We need to store derivatives with respect to q_t (same unit) and the top cell (other unit)
    equation::TervAutoDiffCache
    equation_top_cell::TervAutoDiffCache
    function ControlEquationWell(model, number_of_equations; kwarg...)
        @assert number_of_equations == 1
        alloc = (unit) -> CompactAutoDiffCache(number_of_equations, 1, model, unit = unit; kwarg...)
        # One potential drop per velocity
        target_well = alloc(Well())
        target_topcell = alloc(Cells())
        new(target_well, target_topcell)
    end
end

function associated_unit(::ControlEquationWell) Well() end

function update_equation!(eq::ControlEquationWell, storage, model, dt)
    state = storage.state
    ctrl = state.WellConfiguration.control
    T = ctrl.target
    surf_rate = state.TotalWellMassRate[]
    bhp = state.Pressure[1]
    eq.equation.entries .= well_control_equation(T, surf_rate, value(bhp))
    eq.equation_top_cell.entries .= well_control_equation(T, value(surf_rate), bhp)
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
function select_primary_variables_domain!(S, domain::SingleWellController, system, formulation) 
    S[:TotalWellMassRate] = TotalSurfaceMassRate()
end

function select_equations_domain!(eqs, domain::SingleWellController, system, arg...)
    # eqs[:potential_balance] = (PotentialDropBalanceWell, 1)
    eqs[:control_equation] = (ControlEquationWell, 1)
end

function build_forces(model::SimulationModel{D}; control = nothing, limits = nothing) where {D <: SingleWellController}
    return (control = control, limits = limits,)
end

function initialize_extra_state_fields_domain!(state, model, domain::SingleWellController)
    # Insert structure that holds well control (limits etc) that is then updated before each step
    state[:WellConfiguration] = WellConfiguration()
end

function update_before_step_domain!(storage, model::SimulationModel, domain::SingleWellController, dt, forces)
    # Set control to whatever is on the forces
    storage.state.WellConfiguration.control = forces.control
    storage.state.WellConfiguration.limits = forces.limits
end
