struct UVar <: ScalarVariable end

export VariablePoissonSystem, PoissonSource

Base.@kwdef struct VariablePoissonSystem <: JutulSystem
    time_dependent::Bool = false
end

const PoissonModel = SimulationModel{<:Any, <:VariablePoissonSystem, <:Any, <:Any}

abstract type AbstractPoissonEquation <: JutulEquation end

struct VariablePoissonEquation{T} <: AbstractPoissonEquation
    discretization::T
end

struct VariablePoissonEquationTimeDependent{T} <: AbstractPoissonEquation
    discretization::T
end

function select_equations!(eqs, system::VariablePoissonSystem, model)
    if system.time_dependent
        eq = VariablePoissonEquationTimeDependent(model.domain.discretizations.poisson)
    else
        eq = VariablePoissonEquation(model.domain.discretizations.poisson)
    end
    eqs[:poisson] = eq
end

function select_primary_variables!(S, system::VariablePoissonSystem, model)
    S[:U] = UVar()
end

struct PoissonDiscretization{T} <: JutulDiscretization
    half_face_map::T
    function PoissonDiscretization(g::AbstractJutulMesh)
        N = get_neighborship(g)
        nc = number_of_cells(g)
        hf = half_face_map(N, nc)
        T = typeof(hf)
        return new{T}(hf)
    end
end

function (D::PoissonDiscretization)(i, ::Cells)
    face_map = local_half_face_map(D.half_face_map, i)
    div = F -> local_divergence(F, face_map)
    return (div = div, )
end

struct PoissonFaceCoefficient <: ScalarVariable end

associated_entity(::PoissonFaceCoefficient) = Faces()
default_value(model, ::PoissonFaceCoefficient) = 1.0

function select_parameters!(S, system::VariablePoissonSystem, model)
    S[:K] = PoissonFaceCoefficient()
end

struct PoissonSource <: JutulForce
    cell::Integer
    value::Float64
end

function apply_forces_to_equation!(d, storage, model, eq::AbstractPoissonEquation, eq_s, force::Vector{PoissonSource}, time)
    U = storage.state.U
    for f in force
        c = f.cell
        d[c] -= f.value
    end
end

function setup_forces(model::PoissonModel; sources = nothing)
    return (sources = sources, )
end

function update_equation_in_entity!(eq_buf, self_cell, state, state0, eq::VariablePoissonEquation, model, dt, ldisc = local_discretization(eq, self_cell))
    U = state.U
    K = state.K
    div = ldisc.div
    U_self = state.U[self_cell]
    function flux(other_cell, face, sgn)
        U_other = U[other_cell]
        return K[face]*(U_self - U_other)
    end
    # Equation is just -∇⋅K∇p = 0, or ∇⋅V where V = -K∇p
    eq_buf[] = -div(flux)
end

function update_equation_in_entity!(eq_buf, self_cell, state, state0, eq::VariablePoissonEquationTimeDependent, model, Δt, ldisc = local_discretization(eq, self_cell))
    # Get implicit and explicit variables
    (; U, K) = state
    U0 = state0.U
    # Discretization
    div = ldisc.div
    U_self = state.U[self_cell]
    # Define flux
    function flux(other_cell, face, sgn)
        U_other = U[other_cell]
        return K[face]*(U_self - U_other)
    end
    # Define equation
    ∂U∂t = (U_self - U0[self_cell])/Δt
    eq_buf[] = ∂U∂t - div(flux)
end
