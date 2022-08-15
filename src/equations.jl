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
    # get row and column index in the specific layout we are looking at
    row, col = row_col_sparse(target_entity_index, source_entity_index,
    equation_index, partial_index,
    nentities_target, nentities_source,
    eqs_per_entity, partials_per_entity, layout)
    # find_sparse_position then dispatches on the matrix type to find linear indices
    # for whatever storage format A uses internally
    return find_sparse_position(A, row, col, layout)
end

function find_jac_position(A, target_entity_index, source_entity_index,
    equation_index, partial_index,
    nentities_target, nentities_source,
    eqs_per_entity, partials_per_entity, layout::BlockMajorLayout)

    row = target_entity_index
    col = source_entity_index

    pos = find_sparse_position(A, row, col, layout)
    return (pos-1)*eqs_per_entity*partials_per_entity + eqs_per_entity*(partial_index-1) + equation_index
end

function row_col_sparse(target_entity_index, source_entity_index, # Typically row and column - global index
    equation_index, partial_index,                                # Index of equation and partial derivative - local index
    nentities_target, nentities_source,                           # Row and column sizes for each sub-system
    eqs_per_entity, partials_per_entity,                          # Sizes of the smallest inner system
    layout)
    row = alignment_linear_index(target_entity_index, equation_index, nentities_target, eqs_per_entity, layout)
    col = alignment_linear_index(source_entity_index, partial_index, nentities_source, partials_per_entity, layout)
    return (row, col)
end

function alignment_linear_index(index_outer, index_inner, n_outer, n_inner, ::EquationMajorLayout)
    return n_outer*(index_inner-1) + index_outer
end

function alignment_linear_index(index_outer, index_inner, n_outer, n_inner, ::UnitMajorLayout)
    return n_inner*(index_outer-1) + index_inner
end

function find_sparse_position(A::AbstractSparseMatrix, row, col, layout::JutulMatrixLayout)
    adj = represented_as_adjoint(layout)
    pos = find_sparse_position(A, row, col, adj)
    if pos == 0
        @error "Unable to map cache entry to Jacobian, not allocated in Jacobian matrix." A row col
        error("Jacobian alignment failed. Giving up.")
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

function all_ad_entities(state, states...)
    entities = ad_entities(state)
    for s in states
        e = ad_entities(s)
        merge!(entities, e)
    end
    return entities
end

function setup_equation_storage(model, eq, storage; tag = nothing, kwarg...)
    F!(out, state, state0, i) = update_equation_in_entity!(out, i, state, state0, eq, model, 1.0)
    N = number_of_entities(model, eq)
    n = number_of_equations_per_entity(model, eq)
    e = associated_entity(eq)
    nt = count_active_entities(model.domain, e)
    return create_equation_caches(model, n, N, storage, F!, nt; self_entity = e, kwarg...)
end

function create_equation_caches(model, equations_per_entity, number_of_entities, storage, F!, number_of_entities_total::Integer = 0; self_entity = nothing, kwarg...)
    state = storage[:state]
    state0 = storage[:state0]
    entities = all_ad_entities(state, state0)
    caches = Dict()
    # n = number_of_equations_per_entity(model, eq)
    for (e, epack) in entities
        @timeit "sparsity detection" S = determine_sparsity(F!, equations_per_entity, state, state0, e, entities, number_of_entities)
        number_of_entities_source, T = epack
        has_diagonal = number_of_entities == number_of_entities_total && e == self_entity
        @assert number_of_entities_total > 0 && number_of_entities_source > 0 "nt=$number_of_entities_total ns=$number_of_entities_source"
        @timeit "cache alloc" caches[Symbol(e)] = GenericAutoDiffCache(T, equations_per_entity, e, S, number_of_entities_total, number_of_entities_source, has_diagonal = has_diagonal)
    end
    return convert_to_immutable_storage(caches)
end

"""
Return the domain entity the equation is associated with
"""
function associated_entity(::JutulEquation)
    return Cells()
end

export number_of_equations_per_entity
"""
Get the number of equations per entity. For example, mass balance of two
components will have two equations per grid cell (= entity)
"""
function number_of_equations_per_entity(model::JutulModel, eq::JutulEquation)
    # Default: One equation per entity (= cell,  face, ...)
    @warn "Fixme for $(typeof(eq))"
    error()
    return 1
end

function number_of_equations_per_entity(model::SimulationModel, e::JutulEquation)
    return number_of_equations_per_entity(model.system, e)
end

number_of_equations_per_entity(::JutulSystem, e::JutulEquation) = 1

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
    return number_of_equations_per_entity(model, e)*number_of_entities(model, e)
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
function declare_sparsity(model, e::JutulEquation, eq_storage, entity, layout::EquationMajorLayout)
    primitive = declare_pattern(model, e, eq_storage, entity)
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
        nrow_blocks = number_of_equations_per_entity(model, e)
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

function declare_sparsity(model, e::JutulEquation, eq_storage, entity, layout::BlockMajorLayout)
    primitive = declare_pattern(model, e, eq_storage, entity)
    if isnothing(primitive)
        out = nothing
    else
        I, J = primitive
        n = number_of_equations(model, e)
        m = count_entities(model.domain, entity)

        n_bz = number_of_equations_per_entity(model, e)
        m_bz = degrees_of_freedom_per_entity(model, entity)
        out = SparsePattern(I, J, n, m, layout, n_bz, m_bz)
    end
    return out
end

function declare_sparsity(model, e::JutulEquation, e_storage, entity, layout::UnitMajorLayout)
    primitive = declare_pattern(model, e, e_storage, entity)
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
        nrow_blocks = number_of_equations_per_entity(model, e)
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
function declare_pattern(model, e, eq_s, entity, arg...)
    k = Symbol(entity)
    if haskey(eq_s, k)
        cache = eq_s[k]
        return generic_cache_declare_pattern(cache, arg...)
    else
        out = nothing
    end
    return out
end

function declare_pattern(model, e, eq_storage::CompactAutoDiffCache, entity)
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
is then that of ∂E / ∂P where P are the primary variables of A.
"""
# function declare_cross_model_sparsity(model, other_model, others_equation::JutulEquation)
#    return ([], [])
# end

"""
Update an equation so that it knows where to store its derivatives
in the Jacobian representation.
"""
function align_to_jacobian!(eq_s, eq, jac, model; variable_offset = 0, kwarg...)
    pentities = get_primary_variable_ordered_entities(model)
    for u in pentities
        align_to_jacobian!(eq_s, eq, jac, model, u, variable_offset = variable_offset; kwarg...)
        variable_offset += number_of_degrees_of_freedom(model, u)
    end
end


function align_to_jacobian!(eq_s, eq, jac, model, entity, arg...; context = model.context, positions = nothing, equation_offset = 0, variable_offset = 0, number_of_entities_target = nothing)
    # Use generic version
    k = Symbol(entity)
    has_pos = !isnothing(positions)
    if has_pos
        @assert keys(positions) == keys(eq_s)
    end
    if haskey(eq_s, k)
        cache = eq_s[k]
        if has_pos
            # Align against other positions that is provided
            pos = positions[k]
        else
            # Use positions from cache
            pos = cache.jacobian_positions
        end
        # J = cache.variables
        if isnothing(number_of_entities_target)
            nt = cache.number_of_entities_target
        else
            nt = number_of_entities_target
        end
        I, J = generic_cache_declare_pattern(cache, arg...)
        injective_alignment!(cache, eq, jac, entity, context, pos = pos, target_index = I, source_index = J,
                                                                    number_of_entities_source = cache.number_of_entities_source,
                                                                    number_of_entities_target = nt,
                                                                    target_offset = equation_offset, source_offset = variable_offset)
    else
        @warn "Did not find $k in $(keys(eq_s))"
    end
end

function align_to_jacobian!(eq_s::CompactAutoDiffCache, eq, jac, model, entity; equation_offset = 0, variable_offset = 0)
    if entity == associated_entity(eq)
        # By default we perform a diagonal alignment if we match the associated entity.
        # A diagonal alignment means that the equation for some entity depends only on the values inside that entity.
        # For instance, an equation defined on all Cells will have each entry depend on all values in that Cell.
        diagonal_alignment!(eq_s, eq, jac, entity, model.context, target_offset = equation_offset, source_offset = variable_offset)
    end
end

"""
Update a linearized system based on the values and derivatives in the equation.
"""
function update_linearized_system_equation!(nz, r, model, equation::JutulEquation, diag_cache::CompactAutoDiffCache)
    # NOTE: Default only updates diagonal part
    fill_equation_entries!(nz, r, model, diag_cache)
end

function update_linearized_system_equation!(nz, r, model, equation::JutulEquation, caches)
    for k in keys(caches)
        fill_equation_entries!(nz, r, model, caches[k])
    end
end

"""
Update equation based on currently stored properties
"""
function update_equation!(eq_s, eq::JutulEquation, storage, model, dt)
    state = storage.state
    state0 = storage.state0
    for k in keys(eq_s)
        cache = eq_s[k]
        update_equation_for_entity!(cache, eq, state, state0, model, dt)
    end
end

function update_equation_for_entity!(cache, eq, state, state0, model, dt)
    v = cache.entries
    vars = cache.variables
    for i in 1:number_of_entities(cache)
        ldisc = local_discretization(eq, i)
        @inbounds for j in vrange(cache, i)
            v_i = @views v[:, j]
            state_i = local_ad(state, vars[j], eltype(v))
            state0_i = local_ad(state0, vars[j], eltype(v))
            @inbounds update_equation_in_entity!(v_i, i, state_i, state0_i, eq, model, dt, ldisc)
        end
    end
end

"""
Update an equation with the effect of a force. The default behavior
for any force we do not know about is to assume that the force does
not impact this particular equation.
"""
apply_forces_to_equation!(diag_part, storage, model, eq, eq_s, force, time) = nothing

function convergence_criterion(model, storage, eq::JutulEquation, eq_s, r; dt = 1)
    n = number_of_equations_per_entity(model, eq)
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

@inline function get_diagonal_entries(eq::JutulEquation, eq_s::CompactAutoDiffCache)
    return get_entries(eq_s)
end

@inline function get_diagonal_entries(eq::JutulEquation, eq_s)
    k = Symbol(associated_entity(eq))
    if haskey(eq_s, k)
        cache = eq_s[k]
        D = diagonal_view(cache)
    else
        # Uh oh. Maybe adjoints?
        D = nothing
    end
    return D
end

# @inline function get_diagonal_cache(eq::JutulEquation)
#    return eq.equation
# end
