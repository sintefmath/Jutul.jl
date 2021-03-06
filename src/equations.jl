export allocate_array_ad, get_ad_entity_scalar, update_values!, update_linearized_system_equation!
export value, find_sparse_position



function find_jac_position(A, target_entity_index, source_entity_index, # Typically row and column - global index
    equation_index, partial_index,        # Index of equation and partial derivative - local index
    nentities_target, nentities_source,         # Row and column sizes for each sub-system
    eqs_per_entity, partials_per_entity,      # Sizes of the smallest inner system
    context::JutulContext)
    layout = matrix_layout(context)
    find_jac_position(
        A, target_entity_index, source_entity_index, 
        equation_index, partial_index,
        nentities_target, nentities_source, 
        eqs_per_entity, partials_per_entity, 
        layout
        )
end

function find_jac_position(A, target_entity_index, source_entity_index,
    equation_index, partial_index,
    nentities_target, nentities_source,
    eqs_per_entity, partials_per_entity, layout::JutulMatrixLayout)

    row, col = row_col_sparse(target_entity_index, source_entity_index,
    equation_index, partial_index,
    nentities_target, nentities_source,
    eqs_per_entity, partials_per_entity, layout)
    return find_sparse_position(A, row, col, layout)
end

function find_jac_position(A, target_entity_index, source_entity_index,
    equation_index, partial_index,
    nentities_target, nentities_source,
    eqs_per_entity, partials_per_entity, layout::BlockMajorLayout)
    row, col = row_col_sparse(target_entity_index, source_entity_index,
    equation_index, partial_index,
    nentities_target, nentities_source,
    eqs_per_entity, partials_per_entity, EquationMajorLayout()) # Pass of eqn. major version since we are looking for "scalar" index

    row = target_entity_index
    col = source_entity_index

    pos = find_sparse_position(A, row, col, layout)
    return (pos-1)*eqs_per_entity*partials_per_entity + eqs_per_entity*(partial_index-1) + equation_index
end

function row_col_sparse(target_entity_index, source_entity_index, # Typically row and column - global index
                              equation_index, partial_index,        # Index of equation and partial derivative - local index
                              nentities_target, nentities_source,         # Row and column sizes for each sub-system
                              eqs_per_entity, partials_per_entity,      # Sizes of the smallest inner system
                              layout::EquationMajorLayout)
    row = nentities_target*(equation_index-1) + target_entity_index
    col = nentities_source*(partial_index-1) + source_entity_index
    return (row, col)
end

function row_col_sparse(target_entity_index, source_entity_index, # Typically row and column - global index
    equation_index, partial_index,        # Index of equation and partial derivative - local index
    nentities_target, nentities_source,         # Row and column sizes for each sub-system
    eqs_per_entity, partials_per_entity,      # Sizes of the smallest inner system
    layout::UnitMajorLayout)
    row = eqs_per_entity*(target_entity_index-1) + equation_index
    col = partials_per_entity*(source_entity_index-1) + partial_index
    return (row, col)
end

function find_sparse_position(A::AbstractSparseMatrix, row, col, layout::JutulMatrixLayout)
    adj = represented_as_adjoint(layout)
    pos = find_sparse_position(A, row, col, adj)
    if pos == 0
        @warn "Unable to map $row, $col: Not allocated in matrix."
    end
    return pos
end

function find_sparse_position(A::AbstractSparseMatrix, row, col, is_adjoint)
    if is_adjoint
        a = row
        b = col
    else
        a = col
        b = row
    end
    return find_sparse_position(A, b, a)
end

function find_sparse_position(A::StaticSparsityMatrixCSR, row, col)
    pos = 0
    colval = colvals(A)
    for mat_pos in nzrange(A, row)
        mat_col = colval[mat_pos]
        if mat_col == col
            pos = mat_pos
            break
        end
    end
    return pos
end


function find_sparse_position(A::SparseMatrixCSC, row, col)
    pos = 0
    rowval = rowvals(A)
    for mat_pos in nzrange(A, col)
        mat_row = rowval[mat_pos]
        if mat_row == row
            pos = mat_pos
            break
        end
    end
    return pos
end

function select_equations(domain, system, formulation)
    eqs = OrderedDict{Symbol, Tuple{DataType, Int64}}()
    select_equations_domain!(eqs, domain, system, formulation)
    select_equations_system!(eqs, domain, system, formulation)
    select_equations_formulation!(eqs, domain, system, formulation)
    return eqs
end


function select_equations_domain!(eqs, arg...)
    # Default: No equations
end

function select_equations_system!(eqs, arg...)
    # Default: No equations
end

function select_equations_formulation!(eqs, arg...)
    # Default: No equations
end

"""
Return the domain entity the equation is associated with
"""
function associated_entity(::JutulEquation)
    return Cells()
end

"""
Get the number of equations per entity. For example, mass balance of two
components will have two equations per grid cell (= entity)
"""
function number_of_equations_per_entity(e::JutulEquation)
    # Default: One equation per entity (= cell,  face, ...)
    return get_diagonal_cache(e).equations_per_entity
end

"""
Get the number of entities (e.g. the number of cells) that the equation is defined on.
"""
function number_of_entities(model, e::JutulEquation)
    return count_active_entities(model.domain, associated_entity(e), for_variables = false)
end

"""
Get the total number of equations on the domain of model.
"""
function number_of_equations(model, e::JutulEquation)
    return number_of_equations_per_entity(e)*number_of_entities(model, e)
end

"""
Get the number of partials
"""
function number_of_partials_per_entity(e::JutulEquation)
    return get_diagonal_cache(e).npartials
end

"""
Give out I, J arrays of equal length for a given equation attached
to the given model.
"""
function declare_sparsity(model, e::JutulEquation, entity, layout::EquationMajorLayout)
    primitive = declare_pattern(model, e, entity)
    if isnothing(primitive)
        out = nothing
    else
        I, J = primitive
        # Limit to active set somehow
        ni = length(I)
        nj = length(J)
        if length(I) != length(J)
            error("Pattern I, J for $(typeof(e)) must have equal lengths for entity $(typeof(entity)). (|I| = $ni != $nj = |J|)")
        end
        nu = number_of_entities(model, e)
        nentities = count_active_entities(model.domain, entity, for_variables = false)
        nrow_blocks = number_of_equations_per_entity(e)
        ncol_blocks = number_of_partials_per_entity(model, entity)
        if nrow_blocks > 1
            I = vcat(map((x) -> (x-1)*nu .+ I, 1:nrow_blocks)...)
            J = repeat(J, nrow_blocks)
        end
        if ncol_blocks > 1
            I = repeat(I, ncol_blocks)
            J = vcat(map((x) -> (x-1)*nentities .+ J, 1:ncol_blocks)...)
        end
        n = number_of_equations(model, e)
        m = nentities*ncol_blocks
        out = SparsePattern(I, J, n, m, layout)
    end
    return out
end

function declare_sparsity(model, e::JutulEquation, entity, layout::BlockMajorLayout)
    primitive = declare_pattern(model, e, entity)
    if isnothing(primitive)
        out = nothing
    else
        I, J = primitive
        n = number_of_equations(model, e)
        m = count_entities(model.domain, entity)

        n_bz = number_of_equations_per_entity(e)
        m_bz = degrees_of_freedom_per_entity(model, entity)
        out = SparsePattern(I, J, n, m, layout, n_bz, m_bz)
    end
    return out
end

function declare_sparsity(model, e::JutulEquation, entity, layout::UnitMajorLayout)
    primitive = declare_pattern(model, e, entity)
    if isnothing(primitive)
        out = nothing
    else
        I, J = primitive
        ni = length(I)
        nj = length(J)
        if length(I) != length(J)
            error("Pattern I, J for $(typeof(e)) must have equal lengths for entity $(typeof(entity)). (|I| = $ni != $nj = |J|)")
        end
        nu = number_of_entities(model, e)
        nu_other = count_entities(model.domain, entity)
        nrow_blocks = number_of_equations_per_entity(e)
        ncol_blocks = number_of_partials_per_entity(model, entity)
        nentities = count_entities(model.domain, entity)
        if nrow_blocks > 1
            I = vcat(map((x) -> nrow_blocks*(I .- 1) .+ x, 1:nrow_blocks)...)
            J = repeat(J, nrow_blocks)
        end
        if ncol_blocks > 1
            I = repeat(I, ncol_blocks)
            J = vcat(map((x) -> (J .- 1)*ncol_blocks .+ x, 1:ncol_blocks)...)
        end
        n = number_of_equations(model, e)
        m = nentities*ncol_blocks
        out = SparsePattern(I, J, n, m, layout)
    end
    return out
end

"""
Give out source, target arrays of equal length for a given equation attached
to the given model.
"""
function declare_pattern(model, e, entity)
    if entity == associated_entity(e)
        n = count_entities(model.domain, entity)
        I = collect(1:n)
        return (I, I)
    else
        out = nothing
    end
    return out
end


"""
Give out I, J arrays of equal length for a the impact of a model A on
a given equation E that is attached to some other model B. The sparsity
is then that of ???E / ???P where P are the primary variables of A.
"""
# function declare_cross_model_sparsity(model, other_model, others_equation::JutulEquation)
#    return ([], [])
# end

"""
Update an equation so that it knows where to store its derivatives
in the Jacobian representation.
"""
function align_to_jacobian!(eq::JutulEquation, jac, model; equation_offset = 0, variable_offset = 0)
    pentities = get_primary_variable_ordered_entities(model)
    for u in pentities
        align_to_jacobian!(eq, jac, model, u, equation_offset = equation_offset, variable_offset = variable_offset) 
        variable_offset += number_of_degrees_of_freedom(model, u)
    end
end


function align_to_jacobian!(eq, jac, model, entity; equation_offset = 0, variable_offset = 0)
    if entity == associated_entity(eq)
        # By default we perform a diagonal alignment if we match the associated entity.
        # A diagonal alignment means that the equation for some entity depends only on the values inside that entity.
        # For instance, an equation defined on all Cells will have each entry depend on all values in that Cell.
        diagonal_alignment!(eq.equation, jac, entity, model.context, target_offset = equation_offset, source_offset = variable_offset)
    end
end

"""
Update a linearized system based on the values and derivatives in the equation.
"""

function update_linearized_system_equation!(nz, r, model, equation::JutulEquation)
    # NOTE: Default only updates diagonal part
    fill_equation_entries!(nz, r, model, get_diagonal_cache(equation))
end


"""
Update equation based on currently stored properties
"""
function update_equation!(eq::JutulEquation, storage, model, dt)
    error("No default implementation exists for $(typeof(eq)).")
end

"""
Update an equation with the effect of a force. The default behavior
for any force we do not know about is to assume that the force does
not impact this particular equation.
"""
function apply_forces_to_equation!(storage, model, eq, force, time) end

function convergence_criterion(model, storage, eq::JutulEquation, r; dt = 1)
    n = number_of_equations_per_entity(eq)
    e = zeros(n)
    names = Vector{String}(undef, n)
    for i = 1:n
        @views ri = r[i, :]
        e[i] = norm(ri, Inf)
        names[i] = "R_$i"
    end
    if n == 1
        names = "R"
    end
    R = Dict("AbsMax" => (errors = e, names = names))
    return R
end

@inline function get_diagonal_entries(eq::JutulEquation)
    return get_entries(get_diagonal_cache(eq))
end

@inline function get_diagonal_cache(eq::JutulEquation)
    return eq.equation
end
