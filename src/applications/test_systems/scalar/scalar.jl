export ScalarTestSystem, ScalarTestDomain, ScalarTestForce
export XVar

struct ScalarTestSystem <: JutulSystem end

Base.@kwdef struct ScalarTestDomain <: JutulDomain
    use_manual::Bool = true
end
active_entities(d::ScalarTestDomain, ::Any; kwarg...) = [1]

number_of_cells(::ScalarTestDomain) = 1

#function get_domain_intersection(u::JutulUnit, target_d::ScalarTestDomain, source_d::ScalarTestDomain, target_symbol, source_symbol)
#    # This domain always interacts with the single cell in instances of itself, and nothing else
#    (target = [1], source = [1], target_entity = Cells(), source_entity = Cells())
#end

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
        if domain.use_manual
            D = ManualTestDisc()
        else
            D = AutoTestDisc()
        end
        new{typeof(D)}(D)
    end
end

number_of_equations_per_entity(::ScalarTestEquation) = 1

function select_equations_system!(eqs, domain, system::ScalarTestSystem, formulation)
    eqs[:test_equation] = ScalarTestEquation(domain, system, formulation)
end

function setup_forces(model::SimulationModel{G, S}; sources = nothing) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    return (sources = sources,)
end

struct XVar <: ScalarVariable end

function select_primary_variables!(S, domain, system::ScalarTestSystem, formulation)
    S[:XVar] = XVar()
end

function apply_forces_to_equation!(diag_part, storage, model, eq::ScalarTestEquation, eq_s, force::ScalarTestForce, time)
    @. diag_part -= force.value
end

include("manual.jl")
include("auto.jl")

export ScalarTestCrossTerm
struct ScalarTestCrossTerm <: AdditiveCrossTerm

end

symmetry(::ScalarTestCrossTerm) = CTSkewSymmetry()

# function update_cross_term!(ct::InjectiveCrossTerm, eq::ScalarTestEquation, target_storage, source_storage, target_model, source_model, target, source, dt)
#     X_T = target_storage.state.XVar
#     X_S = source_storage.state.XVar
#     function f(X_S, X_T)
#         X_T - X_S
#     end
#     # Source term with AD context from source model - will end up as off-diagonal block
#     @. ct.crossterm_source = f(X_S, value(X_T))
#     # Source term with AD context from target model - will be inserted into equation
#     @. ct.crossterm_target = f(value(X_S), X_T)
# end

function update_cross_term_in_entity!(out, i, state_t, state0_t, state_s, state0_s, ct::ScalarTestCrossTerm, eq::ScalarTestEquation, dt)
    X_T = only(state_t.XVar)
    X_S = only(state_s.XVar)
    out[] = X_T - X_S
end
