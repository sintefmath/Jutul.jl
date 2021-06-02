export allocate_array_ad, get_ad_unit_scalar, update_values!
export value, find_sparse_position

function find_jac_position(A, target_unit_index, source_unit_index, # Typically row and column - global index
                              equation_index, partial_index,        # Index of equation and partial derivative - local index
                              nunits_target, nunits_source,         # Row and column sizes for each sub-system
                              eqs_per_unit, partials_per_unit,      # Sizes of the smallest inner system
                              layout::EquationMajorLayout)
    row = nunits_target*(equation_index-1) + target_unit_index
    col = nunits_source*(partial_index-1) + source_unit_index
    adj = represented_as_adjoint(layout)
    find_sparse_position(A, row, col, adj)
end

function find_jac_position(A, target_unit_index, source_unit_index, # Typically row and column - global index
    equation_index, partial_index,        # Index of equation and partial derivative - local index
    nunits_target, nunits_source,         # Row and column sizes for each sub-system
    eqs_per_unit, partials_per_unit,      # Sizes of the smallest inner system
    ::UnitMajorLayout)
    error("Not implemented")
end

function find_jac_position(A, target_unit_index, source_unit_index, # Typically row and column - global index
    equation_index, partial_index,        # Index of equation and partial derivative - local index
    nunits_target, nunits_source,         # Row and column sizes for each sub-system
    eqs_per_unit, partials_per_unit,      # Sizes of the smallest inner system
    layout::BlockMajorLayout)
    row = target_unit_index
    col = source_unit_index

    adj = represented_as_adjoint(layout)
    pos = find_sparse_position(A, row, col, adj)
    # We now have the I+J position.
    # We assume that the nzval has eqs_per_unit*partials_per_unit rows,
    # with columns equal to nunits_target * nunits*source
    return (pos-1)*eqs_per_unit*partials_per_unit + partials_per_unit*(equation_index-1) + partial_index
end

function find_sparse_position(A::SparseMatrixCSC, row, col, is_adjoint)
    if is_adjoint
        a = row
        b = col
    else
        a = col
        b = row

    end
    find_sparse_position(A, b, a)
end

function find_sparse_position(A::SparseMatrixCSC, row, col)
    for pos = A.colptr[col]:A.colptr[col+1]-1
        if A.rowval[pos] == row
            return pos
        end
    end
    return 0
end

function select_equations(domain, system, formulation)
    eqs = OrderedDict()
    select_equations!(eqs, domain, system, formulation)
    return eqs
end

function select_equations!(eqs, domain, system, formulation)
    select_equations!(eqs, system)
end

function select_equations!(eqs, system)
    # Default: No equations
end

"""
Return the domain unit the equation is associated with
"""
function associated_unit(::TervEquation)
    return Cells()
end

"""
Get the number of equations per unit. For example, mass balance of two
components will have two equations per grid cell (= unit)
"""
function number_of_equations_per_unit(e::TervEquation)
    # Default: One equation per unit (= cell,  face, ...)
    return get_diagonal_cache(e).equations_per_unit
end

"""
Get the number of units (e.g. the number of cells) that the equation is defined on.
"""
function number_of_units(model, e::TervEquation)
    return count_units(model.domain, associated_unit(e))
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
    return get_diagonal_cache(e).npartials
end

"""
Give out I, J arrays of equal length for a given equation attached
to the given model.
"""
function declare_sparsity(model, e::TervEquation, unit, layout::EquationMajorLayout)
    primitive = declare_pattern(model, e, unit)
    if isnothing(primitive)
        out = nothing
    else
        I = primitive[1]
        J = primitive[2]
        nu = primitive[3]
        nrow_blocks = number_of_equations_per_unit(e)
        ncol_blocks = number_of_partials_per_unit(model, unit)
        nunits = count_units(model.domain, unit)
        if nrow_blocks > 1
            I = vcat(map((x) -> (x-1)*nu .+ I, 1:nrow_blocks)...)
            J = repeat(J, nrow_blocks)
        end
        if ncol_blocks > 1
            I = repeat(I, ncol_blocks)
            J = vcat(map((x) -> (x-1)*nu .+ J, 1:ncol_blocks)...)
        end
        n = number_of_equations(model, e)
        m = nunits*ncol_blocks
        out = (I, J, n, m)
    end
    return out
end

function declare_sparsity(model, e::TervEquation, unit, ::BlockMajorLayout)
    declare_pattern(model, e, unit)
end

"""
Give out source, target arrays of equal length for a given equation attached
to the given model.
"""
function declare_pattern(model, e::TervEquation, unit)
    return nothing
end

function declare_pattern(model, e::DiagonalEquation, unit)
    if unit == associated_unit(e)
        n = count_units(model.domain, unit)
        I = collect(1:n)
        return (I, I, n, n)
    else
        out = nothing
    end
    return out
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
function align_to_jacobian!(eq::TervEquation, jac, model; equation_offset = 0, variable_offset = 0)
    punits = get_primary_variable_ordered_units(model)
    for u in punits
        align_to_jacobian!(eq, jac, model, u, equation_offset = equation_offset, variable_offset = variable_offset) 
        variable_offset += number_of_degrees_of_freedom(model, u)
    end
end

function align_to_jacobian!(::TervEquation, jac, model, unit; equation_offset = 0, variable_offset = 0) end


function align_to_jacobian!(eq::DiagonalEquation, jac, model, unit; equation_offset = 0, variable_offset = 0)
    if unit == associated_unit(eq)
        layout = matrix_layout(model.context)
        diagonal_alignment!(eq.equation, jac, unit, layout, target_offset = equation_offset, source_offset = variable_offset)
    end
end

"""
Update a linearized system based on the values and derivatives in the equation.
"""

function update_linearized_system_equation!(nz, r, model, equation::TervEquation)
    # NOTE: Default only updates diagonal part
    fill_equation_entries!(nz, r, model, get_diagonal_cache(equation))
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

@inline function get_diagonal_part(eq::TervEquation)
    return get_entries(get_diagonal_cache(eq))
end

@inline function get_diagonal_cache(eq::TervEquation)
    return eq.equation
end
