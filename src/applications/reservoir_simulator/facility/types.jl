export TotalMassVelocityMassFractionsFlow

abstract type FacilitySystem <: TervSystem end
struct PredictionMode <: FacilitySystem end
struct HistoryMode <: FacilitySystem end

abstract type SurfaceFacilityDomain <: TervDomain end
abstract type WellControllerDomain <: SurfaceFacilityDomain end
struct WellGroup <: WellControllerDomain
    well_symbols::Vector{Symbol}
end

struct Wells <: TervUnit end
struct TotalSurfaceMassRate <: ScalarVariable end
abstract type WellTarget end

struct BottomHolePressureTarget <: WellTarget
    value::AbstractFloat
end

struct SinglePhaseRateTarget <: WellTarget
    value::AbstractFloat
    phase::AbstractPhase
end

struct DisabledTarget <: WellTarget end
abstract type WellForce <: TervForce end
abstract type WellControlForce <: WellForce end

struct DisabledControl <: WellControlForce
    target::DisabledTarget
    function DisabledControl()
        t = DisabledTarget()
        new(t)
    end
end

struct InjectorControl <: WellControlForce
    target::WellTarget
    injection_mixture
    function InjectorControl(target, mix)
        if isa(mix, Real)
            mix = [mix]
        end
        mix = vec(mix)
        @assert sum(mix) ≈ 1
        new(target, mix)
    end
end

struct ProducerControl <: WellControlForce
    target::WellTarget
end

struct WellGroupConfiguration
    control
    limits
    function WellGroupConfiguration(well_symbols, control = nothing, limits = nothing)
        if isnothing(control)
            control = OrderedDict{Symbol, WellControlForce}()
            for s in well_symbols
                control[s] = DisabledControl()
            end
        end
        if isnothing(limits)
            limits = OrderedDict{Symbol, Any}()
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
    # We need to store derivatives with respect to q_t (same entity) and the top cell (other entity)
    equation::TervAutoDiffCache
    function ControlEquationWell(model, number_of_equations; kwarg...)
        # @assert number_of_equations == 1
        nw = count_entities(model.domain, Wells())
        alloc = (entity) -> CompactAutoDiffCache(number_of_equations, nw, model, entity = entity; kwarg...)
        # One potential drop per velocity
        target_well = alloc(Wells())
        new(target_well)
    end
end

struct TotalMassVelocityMassFractionsFlow <: FlowType end
