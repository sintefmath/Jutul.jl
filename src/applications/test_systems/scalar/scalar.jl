export ScalarTestSystem, ScalarTestDomain, ScalarTestForce
export XVar

struct ScalarTestSystem <: JutulSystem end

Base.@kwdef struct ScalarTestDomain <: JutulDomain
    use_manual::Bool = true
end
active_entities(d::ScalarTestDomain, ::Any; kwarg...) = [1]

function declare_entities(G::ScalarTestDomain)
    return [(entity = Cells(), count = 1)]
end

number_of_cells(::ScalarTestDomain) = 1

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
    function ScalarTestEquation(model)
        if model.domain.use_manual
            D = ManualTestDisc()
        else
            D = AutoTestDisc()
        end
        new{typeof(D)}(D)
    end
end

number_of_equations_per_entity(model::SimulationModel, ::ScalarTestEquation) = 1

function select_equations!(eqs, system::ScalarTestSystem, model::SimulationModel)
    eqs[:test_equation] = ScalarTestEquation(model)
end

function setup_forces(model::SimulationModel{G, S}; sources = nothing) where {G<:ScalarTestDomain, S<:ScalarTestSystem}
    return (sources = sources,)
end

struct XVar <: ScalarVariable end

function select_primary_variables!(S, system::ScalarTestSystem, model::SimulationModel)
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

function update_cross_term_in_entity!(out, i, state_t, state0_t,
                                              state_s, state0_s, 
                                              model_t, model_s,
                                              ct::ScalarTestCrossTerm, eq::ScalarTestEquation, dt, ldisc = local_discretization(ct, i))
    X_T = only(state_t.XVar)
    X_S = only(state_s.XVar)
    out[] = X_T - X_S
end
