export allocate_array_ad, get_ad_entity_scalar, update_values!, update_linearized_system_equation!
export value, find_sparse_position

function find_jac_position(A,
        target_entity_index, source_entity_index, # Typically row and column - global index
        row_offset, column_offset,
        target_entity_offset, source_entity_offset,
        equation_index, partial_index,            # Index of equation and partial derivative - local index
        nentities_target, nentities_source,       # Row and column sizes for each sub-system
        eqs_per_entity, partials_per_entity,      # Sizes of the smallest inner system
        context::JutulContext;
        number_of_equations_for_entity = eqs_per_entity
    )
    layout = matrix_layout(context)
    return find_jac_position(
        A, target_entity_index, source_entity_index,
        row_offset, column_offset,
        target_entity_offset,source_entity_offset,
        equation_index, partial_index,
        nentities_target, nentities_source,
        eqs_per_entity, partials_per_entity,
        layout, layout,
        number_of_equations_for_entity = number_of_equations_for_entity
        )
end

function find_jac_position(A,
        target_entity_index, source_entity_index,
        row_offset, column_offset,
        target_entity_offset, source_entity_offset,
        equation_index, partial_index,
        nentities_target, nentities_source,
        eqs_per_entity, partials_per_entity,
        row_layout, col_layout;
        number_of_equations_for_entity = eqs_per_entity
    )
    # get row and column index in the specific layout we are looking at
    if row_layout isa BlockMajorLayout
        b = (target_entity_offset ÷ nentities_target)
        equation_index += b
        target_entity_offset = 0
        # The block we are in could actually be bigger. This only matters for
        # entity/block major stuff.
        eqs_per_entity = max(eqs_per_entity, partials_per_entity)
    end
    row_layout = scalarize_layout(row_layout, col_layout)
    col_layout = scalarize_layout(col_layout, row_layout)

    row, col = row_col_sparse(
        target_entity_index, source_entity_index,
        target_entity_offset, source_entity_offset,
        equation_index, partial_index,
        nentities_target, nentities_source,
        eqs_per_entity, partials_per_entity,
        row_layout, col_layout
    )
    # find_sparse_position then dispatches on the matrix type to find linear indices
    # for whatever storage format A uses internally
    return find_sparse_position(A, row + row_offset, col + column_offset, row_layout)
end

function find_jac_position(
        A,
        target_entity_index, source_entity_index,
        row_offset, column_offset,
        target_entity_offset, source_entity_offset,
        equation_index, partial_index,
        nentities_target, nentities_source,
        eqs_per_entity, partials_per_entity,
        row_layout::T, col_layout::T;
        number_of_equations_for_entity = eqs_per_entity
    ) where T<:BlockMajorLayout

    N = partials_per_entity
    if eqs_per_entity < partials_per_entity
        if target_entity_offset != 0
            # This happens when we have a block layout with several equation
            # groups making up the block but some of the groups have fewer
            # equations than the number of partials per entity. In that case, we
            # need to adjust the offsets and indices to make sure we are looking
            # at the right block. TODO: This needs some double checking...
            n = fld(target_entity_offset, nentities_target)
            # Now we are really inside the block! Reset the target entity offset
            # and adjust the equation index to point to the right place inside
            # the block.
            target_entity_offset = 0
            equation_index = equation_index + n
        end
    end

    row_base = row_offset
    col_base = column_offset

    row_base = row_base ÷ N
    col_base = col_base ÷ N

    row = target_entity_offset + target_entity_index + row_base
    col = source_entity_offset + source_entity_index + col_base
    inner_layout = EntityMajorLayout()

    adjoint_layout = represented_as_adjoint(row_layout)
    block_matrix_length = N*N
    if adjoint_layout
        @assert represented_as_adjoint(col_layout)
        pos = find_sparse_position(A, row, col, inner_layout)
        base_ix = (pos-1)*block_matrix_length
        # TODO: Check this.
        ix = base_ix + N*(equation_index-1) + partial_index# + offset
    else
        pos = find_sparse_position(A, row, col, inner_layout)
        base_ix = (pos-1)*block_matrix_length
        ix = base_ix + N*(partial_index-1) + equation_index# + offset
    end

    return ix
end

function row_col_sparse(
        target_entity_index, source_entity_index,   # Typically row and column - global index
        target_entity_offset, source_entity_offset,
        equation_index, partial_index,              # Index of equation and partial derivative - local index
        nentities_target, nentities_source,         # Row and column sizes for each sub-system
        eqs_per_entity, partials_per_entity,        # Sizes of the smallest inner system
        row_layout, col_layout
    )
    row = alignment_linear_index(target_entity_index + target_entity_offset, equation_index, nentities_target, eqs_per_entity, row_layout)
    col = alignment_linear_index(source_entity_index + source_entity_offset, partial_index, nentities_source, partials_per_entity, col_layout)
    return (row, col)
end

function alignment_linear_index(index_outer, index_inner, n_outer, n_inner, ::EquationMajorLayout)
    return n_outer*(index_inner-1) + index_outer
end

function alignment_linear_index(index_outer, index_inner, n_outer, n_inner, ::Union{EntityMajorLayout, BlockMajorLayout})
    return n_inner*(index_outer-1) + index_inner
end

function find_sparse_position(A::AbstractSparseMatrix, row, col, layout::JutulMatrixLayout)
    adj = represented_as_adjoint(layout)
    pos = find_sparse_position(A, row, col, adj)
    if pos == 0
        I, J = findnz(A)
        IJ = map((i, j) -> (i, j), I, J)
        @error "Unable to map cache entry to Jacobian, ($row,$col) not allocated in Jacobian matrix." A row col represented_as_adjoint(layout) IJ
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

function create_equation_caches(model, equations_per_entity, number_of_entities, storage, F!, number_of_entities_total::Integer = 0;
        global_map = global_map(model),
        self_entity = nothing,
        extra_sparsity = nothing,
        ad = true,
        kwarg...
    )
    state = storage[:state]
    state0 = storage[:state0]
    caches = Dict()
    self_entity_found = false
    if ad
        entities = all_ad_entities(state, state0)
        for (e, epack) in entities
            is_self = e == self_entity
            self_entity_found = self_entity_found || is_self
            @tic "sparsity detection" S = determine_sparsity(F!, equations_per_entity, state, state0, e, entities, number_of_entities)
            if !isnothing(extra_sparsity)
                # We have some extra sparsity, need to merge that in
                S_e = extra_sparsity[entity_as_symbol(e)]
                @assert length(S_e) == length(S)
                for (i, s_extra) in enumerate(S_e)
                    for extra_ind in s_extra
                        push!(S[i], extra_ind)
                    end
                    unique!(S[i])
                end
            end
            _, T = epack
            S, number_of_entities_source = remap_sparsity!(S, e, model)
            has_diagonal = number_of_entities == number_of_entities_total && is_self
            @assert number_of_entities_total > 0 && number_of_entities_source > 0 "nt=$number_of_entities_total ns=$number_of_entities_source for $T"
            @tic "cache alloc" cache = GenericAutoDiffCache(T, equations_per_entity, e, S, number_of_entities_total, number_of_entities_source, has_diagonal = has_diagonal, global_map = global_map)
            caches[entity_as_symbol(e)] = cache
        end
    end
    if !self_entity_found
        caches[:numeric] = zeros(equations_per_entity, number_of_entities)
    end
    return convert_to_immutable_storage(caches)
end

@inline function entity_as_symbol(::T) where T<:JutulEntity
    return Symbol(T.name.name)::Symbol
end

function remap_sparsity!(S, var_entity, eq_model)
    # Filter away inactive entities in sparsity
    d_e = eq_model.domain
    map_e = global_map(d_e)
    ns = count_active_entities(d_e, var_entity)
    if !(map_e isa TrivialGlobalMap)
        map_e = global_map(d_e)
        active_vars = active_entities(d_e, map_e, var_entity, for_variables = false)
        n = length(S)
        for i in 1:n
            # Alter the set - renumber and remove entries that are not a part of local set
            s = S[i]
            filter!(x -> x in active_vars, s)
            for j in eachindex(s)
                # s[j] = index_map(s[j], map_e, VariableSet(), EquationSet(), var_entity)
            end
        end
    end
    return (S, ns)
end

"""
Return the domain entity the equation is associated with
"""
function associated_entity(::JutulEquation)
    return Cells()
end

export number_of_equations_per_entity
"""
    n = number_of_equations_per_entity(model::JutulModel, eq::JutulEquation)

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

function number_of_equations(model)
    n = 0
    for eq in values(model.equations)
        n += number_of_equations(model, eq)
    end
    return n
end

"""
Give out I, J arrays of equal length for a given equation attached
to the given model.
"""
function declare_sparsity(model, e, eq_storage, entity, row_layout, col_layout = row_layout)
    primitive = declare_pattern(model, e, eq_storage, entity)
    if isnothing(primitive)
        out = nothing
    else
        I, J = primitive
        # Limit to active set
        ni = length(I)
        nj = length(J)
        if length(I) != length(J)
            error("Pattern I, J for $(typeof(e)) must have equal lengths for entity $(typeof(entity)). (|I| = $ni != $nj = |J|)")
        end
        # Rows
        I, J, n = row_expansion(I, J, model, e, entity, row_layout)
        # Columns
        I, J, m = column_expansion(I, J, model, e, entity, col_layout)
        out = SparsePattern(I, J, n, m, row_layout, col_layout)
    end
    return out
end

function row_expansion(I, J, model, e, entity, row_layout)
    nu = number_of_entities(model, e)
    nrow_blocks = number_of_equations_per_entity(model, e)
    n_eqs = number_of_equations(model, e)
    # Rows
    I, J = expand_block_indices(I, J, nu, nrow_blocks, row_layout)
    return (I, J, n_eqs)
end

function column_expansion(I, J, model, e, entity, col_layout)
    ncol_blocks = number_of_partials_per_entity(model, entity)
    n_entity = count_active_entities(model.domain, entity, for_variables = false)
    # (switched order)
    m = n_entity*ncol_blocks
    J, I = expand_block_indices(J, I, n_entity, ncol_blocks, col_layout)
    return (I, J, m)
end

function expand_block_indices(I, J, ntotal, neqs, layout::EquationMajorLayout; equation_offset = 0, block_size = neqs)
    if neqs > 1
        I = vcat(map((x) -> (x-1)*ntotal .+ I, 1:neqs)...)
        J = repeat(J, neqs)
    end
    return (I, J)
end

function expand_block_indices(I, J, ntotal, neqs, layout::EntityMajorLayout; equation_offset = 0, block_size = neqs)
    T = eltype(I)
    n = length(I)
    @assert length(J) == n
    I_expand = T[]
    J_expand = T[]
    for (i, j) in zip(I, J)
        for eq in 1:neqs
            ii = block_size*(i - 1) + equation_offset + eq
            push!(I_expand, ii)
            push!(J_expand, j)
        end
    end
    @assert length(I_expand) == length(J_expand)
    # I = vcat(map((x) -> nblocks*(I .- 1 .+ equation_offset) .+ x, (1+equation_offset):nblocks)...)
    # J = repeat(J, nblocks)
    return (I_expand, J_expand)
end


function declare_sparsity(model, e, eq_storage, entity, row_layout::T, col_layout::T = row_layout) where T<:BlockMajorLayout
    primitive = declare_pattern(model, e, eq_storage, entity)
    if isnothing(primitive)
        out = nothing
    else
        I, J = primitive
        n = number_of_equations(model, e)
        m = count_entities(model.domain, entity)

        n_bz = number_of_equations_per_entity(model, e)
        m_bz = degrees_of_freedom_per_entity(model, entity)
        out = SparsePattern(I, J, n, m, row_layout, col_layout, n_bz, m_bz)
    end
    return out
end

"""
Give out source, target arrays of equal length for a given equation attached
to the given model.
"""
function declare_pattern(model, e, eq_s, entity, arg...)
    k = entity_as_symbol(entity)
    if haskey(eq_s, k)
        cache = eq_s[k]
        out = generic_cache_declare_pattern(cache, arg...)
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
    variable_offset
end


function align_to_jacobian!(eq_s, eq, jac, model, entity, arg...;
        context = model.context,
        positions = nothing,
        row_offset = 0,
        column_offset = 0,
        equation_offset = 0,
        variable_offset = 0,
        number_of_entities_target = nothing,
        kwarg...
    )
    # Use generic version
    k = entity_as_symbol(entity)
    has_pos = !isnothing(positions)
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
        injective_alignment!(cache, eq, jac, entity, context,
            pos = pos,
            target_index = I,
            source_index = J,
            number_of_entities_source = cache.number_of_entities_source,
            number_of_entities_target = nt,
            row_offset = row_offset,
            column_offset = column_offset,
            target_offset = equation_offset,
            source_offset = variable_offset
            ; kwarg...)
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
function update_linearized_system_equation!(nz::AbstractArray, r, model, equation::JutulEquation, diag_cache::CompactAutoDiffCache)
    # NOTE: Default only updates diagonal part
    fill_equation_entries!(nz, r, model, diag_cache)
end

function update_linearized_system_equation!(nz, r, model, equation::JutulEquation, caches)
    for k in keys(caches)
        if k == :numeric
            continue
        end
        fill_equation_entries!(nz, r, model, caches[k])
    end
end

function update_linearized_system_equation!(nz::Missing, r, model, equation::JutulEquation, cache)
    d = get_diagonal_entries(equation, cache)
    for i in eachindex(d, r)
        r[i] = d[i]
    end
    return r
end

"""
Update equation based on currently stored properties
"""
function update_equation!(eq_s, eq::JutulEquation, storage, model, dt)
    state = storage.state
    state0 = storage.state0
    for i in 1:number_of_entities(model, eq)
        prepare_equation_in_entity!(i, eq, eq_s, state, state0, model, dt)
    end
    if eq_s isa AbstractArray
        update_equation_for_entity!(eq_s, eq, state, state0, model, dt)
    else
        for k in keys(eq_s)
            if k == :numeric
                continue
            end
            cache = eq_s[k]
            update_equation_for_entity!(cache, eq, state, state0, model, dt)
        end
    end
end

@inline prepare_equation_in_entity!(i, eq, eq_s, state, state0, model, dt) = nothing

function update_equation_for_entity!(cache, eq, state, state0, model, dt)
    T = eltype(cache.entries)
    local_state = local_ad(state, 1, T)
    local_state0 = local_ad(state0, 1, T)
    inner_update_equation_for_entity(cache, eq, local_state, local_state0, model, dt)
end

function update_equation_for_entity!(cache::AbstractMatrix, eq, state, state0, model, dt)
    for i in axes(cache, 2)
        ldisc = local_discretization(eq, i)
        v_i = view(cache, :, i)
        update_equation_in_entity!(v_i, i, state, state0, eq, model, dt, ldisc)
    end
    return cache
end

function inner_update_equation_for_entity(cache, eq, state, state0, model, dt)
    v = cache.entries
    vars = cache.variables

    ne = number_of_entities(cache)
    tb = minbatch(model.context, ne)
    @batch minbatch=tb for i in 1:ne
        ldisc = local_discretization(eq, i)
        @inbounds for j in vrange(cache, i)
            v_i = @views v[:, j]
            var = vars[j]
            state_i = new_entity_index(state, var)
            state0_i = new_entity_index(state0, var)
            @inbounds update_equation_in_entity!(v_i, i, state_i, state0_i, eq, model, dt, ldisc)
        end
    end
end

"""
    apply_forces_to_equation!(diag_part, storage, model, eq, eq_s, force, time)

Update an equation with the effect of a force. The default behavior
for any force we do not know about is to assume that the force does
not impact this particular equation.
"""
function apply_forces_to_equation!(diag_part, storage, model, eq, eq_s, force, time)
    nothing
end

"""
    convergence_criterion(model, storage, eq, eq_s, r; dt = 1)

Get the convergence criterion values for a given equation. Can be checked against the corresponding tolerances.

# Arguments
- `model`: model that generated the current equation.
- `storage`: global simulator storage.
- `eq::JutulEquation`: equation implementation currently being checked
- `eq_s`: storage for `eq` where values are contained.
- `r`: the local residual part corresponding to this model, as a matrix with column index equaling entity index
"""
function convergence_criterion(model, storage, eq::JutulEquation, eq_s, r; dt = 1.0, update_report = missing)
    n = number_of_equations_per_entity(model, eq)
    @tic "default" @tullio max e[i] := abs(r[i, j])
    if n == 1
        names = "R"
    else
        names = map(i -> "R_$i", 1:n)
    end
    R = (AbsMax = (errors = e, names = names), )
    return R
end

@inline function get_diagonal_entries(eq::JutulEquation, eq_s::CompactAutoDiffCache)
    return get_entries(eq_s)
end

"""
    get_diagonal_entries(eq::JutulEquation, eq_s)

Get the diagonal entries of a cache, i.e. the entries where entity type and index equals that of the governing equation.

Note: Be very careful about modifications to this array, as this is a view into the internal AD buffers and it is very easy
to create inconsistent Jacobians.
"""
@inline function get_diagonal_entries(eq::JutulEquation, eq_s)
    return get_diagonal_entries(eq, eq_s, associated_entity(eq))
end

@inline function get_diagonal_entries(eq::JutulEquation, eq_s, e)
    k = entity_as_symbol(e)
    if haskey(eq_s, k)
        cache = eq_s[k]
        D = diagonal_view(cache)
    elseif haskey(eq_s, :numeric)
        D = eq_s[:numeric]
    else
        # Uh oh. Maybe adjoints?
        D = nothing
    end
    return D
end

@inline function get_diagonal_entries(eq::JutulEquation, eq_s::AbstractArray)
    return eq_s
end

function transfer_accumulation!(acc, eq::ConservationLaw, state)
    s = Jutul.conserved_symbol(eq)
    @. acc = state[s]
end

function transfer_accumulation!(acc, eq::JutulEquation, state)
    @. acc = zero(eltype(acc))
end
