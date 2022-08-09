struct UVar <: ScalarVariable end

export VariablePoissonSystem

struct VariablePoissonSystem <: JutulSystem end

struct VariablePoissonEquation{T} <: JutulEquation
    discretization::T
end

function select_equations!(eqs, system::VariablePoissonSystem, model)
    eqs[:poisson] = VariablePoissonEquation(model.domain.discretizations.poisson)
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

function Jutul.update_equation_in_entity!(eq_buf, self_cell, state, state0, eq::VariablePoissonEquation, model, dt, ldisc = local_discretization(eq, self_cell))
    U = state.U
    K = state.K
    div = ldisc.div
    U_self = state.U[self_cell]
    function flux(other_cell, face, sgn)
        U_other = U[other_cell]
        return K[face]*(U_self - U_other)
    end
    # Equation is just -∇⋅K∇p = 0, or ∇⋅V where V = -K∇p
    eq = -div(flux)
    if self_cell == 1
        eq += 1
    elseif self_cell == number_of_cells(model.domain.grid)
        eq -= 1
    end
    @info self_cell eq
    eq_buf[] = eq
end
