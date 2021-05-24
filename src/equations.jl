export allocate_array_ad, get_ad_unit_scalar, update_values!
export value, find_sparse_position

function find_sparse_position(A::SparseMatrixCSC, row, col)
    for pos = A.colptr[col]:A.colptr[col+1]-1
        if A.rowval[pos] == row
            return pos
        end
    end
    return 0
end


"""
Return the domain unit the equation is associated with
"""
function domain_unit(::TervEquation)
    return Cells()
end

"""
Get the number of equations per unit. For example, mass balance of two
components will have two equations per grid cell (= unit)
"""
function number_of_equations_per_unit(::TervEquation)
    # Default: One equation per unit (= cell,  face, ...)
    return 1
end

"""
Get the number of units (e.g. the number of cells) that the equation is defined on.
"""
function number_of_units(model, e::TervEquation)
    return count_units(model.domain, domain_unit(e))
end

"""
Get the total number of equations on the domain of model.
"""
function number_of_equations(model, e::TervEquation)
    return number_of_equations_per_unit(e)*number_of_units(model, e)
end

"""
Get the number of partials
"""
function number_of_partials_per_unit(e::TervEquation)
    return length(e.equation[1].partials)
end

"""
Give out I, J arrays of equal length for a given equation attached
to the given model.
"""
function declare_sparsity(model, e::TervEquation)
    return ([], [], NaN, NaN)
end

"""
Give out I, J arrays of equal length for a the impact of a model A on
a given equation E that is attached to some other model B. The sparsity
is then that of ∂E / ∂P where P are the primary variables of A.
"""
# function declare_cross_model_sparsity(model, other_model, others_equation::TervEquation)
#    return ([], [])
# end

"""
Update an equation so that it knows where to store its derivatives
in the Jacobian representation.
"""
function align_to_jacobian!(::TervEquation, jac, model) end


"""
Update a linearized system based on the values and derivatives in the equation.
"""
function update_linearized_system_subset!(lsys, model, equation::TervEquation; r_subset = nothing)
    if isnothing(r_subset)
        r = lsys.r
    else
        r = view(lsys.r, r_subset)
    end
    jac = lsys.jac
    update_linearized_system_subset!(jac, r, model, equation)
end

function update_linearized_system_subset!(jac, r, model, equation::TervEquation)
    e = equation.equation
    e::TervAutoDiffCache
    update_linearized_system_subset!(jac, r, model, e)
end


"""
Update equation based on currently stored properties
"""
function update_equation!(eqs::TervEquation, storage, model, dt)

end

"""
Update an equation with the effect of a force. The default behavior
for any force we do not know about is to assume that the force does
not impact this particular equation.
"""
function apply_forces_to_equation!(storage, model, eq, force) end


function convergence_criterion(model, storage, eq::TervEquation, r; dt = 1)
    n = number_of_equations_per_unit(eq)
    m = length(r) ÷ n
    e = zeros(n)
    for i = 1:n
        x = view(r, ((i-1)*m+1):(i*m))
        e[i] = norm(x, Inf)
    end
    return (e, 1.0)
end

function get_diagonal_part(eq::TervEquation)
    return eq.equation
end
