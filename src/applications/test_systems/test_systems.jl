export ScalarTestSystem, ScalarTestDomain, ScalarTestForce
export XVar

struct ScalarTestSystem <: JutulSystem end

Base.@kwdef struct ScalarTestDomain <: JutulDomain
    use_manual::Bool = true
end
active_entities(d::ScalarTestDomain, ::Any; kwarg...) = [1]

function number_of_cells(::ScalarTestDomain) 1 end

function get_domain_intersection(u::JutulUnit, target_d::ScalarTestDomain, source_d::ScalarTestDomain, target_symbol, source_symbol)
    # This domain always interacts with the single cell in instances of itself, and nothing else
    (target = [1], source = [1], target_entity = Cells(), source_entity = Cells())
end

# Driving force for the test equation
struct ScalarTestForce
    value
end

abstract type AbstractTestDisc <: JutulDiscretization end
struct ManualTestDisc <: AbstractTestDisc end
struct AutoTestDisc <: AbstractTestDisc end

# Equations
struct ScalarTestEquation{D} <: DiagonalEquation
    discretization::D
    function ScalarTestEquation(domain, system, formulation)
        # Ω = model.domain
        # nc = number_of_cells(Ω)
        if domain.use_manual
            D = ManualTestDisc()
        else
            D = AutoTestDisc()
        end
        # @assert nc == 1 # We use nc for clarity of the interface - but it should always be one!
        # ne = 1 # Single, scalar equation
        # e = CompactAutoDiffCache(ne, nc, npartials, context = context; kwarg...)
        new{typeof(D)}(D)
    end
end

number_of_equations_per_entity(::ScalarTestEquation) = 1

function setup_equation_storage(model, e::ScalarTestEquation{ManualTestDisc}; kwarg...)
    Ω = model.domain
    nc = number_of_cells(Ω)
    @assert nc == 1 # We use nc for clarity of the interface - but it should always be one!
    ne = 1 # Single, scalar equation
    npartials = number_of_equations_per_entity(e)
    e = CompactAutoDiffCache(ne, nc, npartials, context = model.context; kwarg...)
    return e
end

function declare_pattern(model, e::ScalarTestEquation{ManualTestDisc}, eq_storage, unit)
    @assert unit == Cells()
    return ([1], [1])
end

function select_equations_system!(eqs, domain, system::ScalarTestSystem, formulation)
    eqs[:test_equation] = ScalarTestEquation(domain, system, formulation)
end

function update_equation!(eq_s::CompactAutoDiffCache, eq::ScalarTestEquation, storage, model, dt)
    X = storage.state.XVar
    X0 = storage.state0.XVar
    equation = get_entries(eq_s)
    @. equation = (X - X0)/dt
end

function setup_forces(model::SimulationModel{G, S}; sources = nothing) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    return (sources = sources,)
end

function apply_forces_to_equation!(storage, model, eq::ScalarTestEquation, eq_s, force::ScalarTestForce, time)
    equation = get_entries(eq_s)
    @. equation -= force.value
end

function update_cross_term!(ct::InjectiveCrossTerm, eq::ScalarTestEquation, target_storage, source_storage, target_model, source_model, target, source, dt)
    X_T = target_storage.state.XVar
    X_S = source_storage.state.XVar
    function f(X_S, X_T)
        X_T - X_S
    end
    # Source term with AD context from source model - will end up as off-diagonal block
    @. ct.crossterm_source = f(X_S, value(X_T))
    # Source term with AD context from target model - will be inserted into equation
    @. ct.crossterm_target = f(value(X_S), X_T)
end

struct XVar <: ScalarVariable end

function select_primary_variables!(S, domain, system::ScalarTestSystem, formulation)
    S[:XVar] = XVar()
end
