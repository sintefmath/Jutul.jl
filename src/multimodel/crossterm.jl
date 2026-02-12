local_discretization(::CrossTerm, i) = nothing

function declare_sparsity(target_model, source_model, eq, x::CrossTerm, x_storage, entity_indices, target_entity, source_entity, row_layout, col_layout; equation_offset = 0, block_size = 1)
    primitive = declare_pattern(target_model, x, x_storage, source_entity, entity_indices)
    if isnothing(primitive)
        out = nothing
    else
        target_impact = primitive[1]
        source_impact = primitive[2]
        nentities_source = count_active_entities(source_model.domain, source_entity)
        nentities_target = count_active_entities(target_model.domain, target_entity)

        n_partials = number_of_partials_per_entity(source_model, source_entity)
        n_eqs = number_of_equations_per_entity(target_model, eq)
        out = inner_sparsity_ct(
            target_impact, source_impact,
            nentities_source, nentities_target,
            n_partials, n_eqs,
            row_layout, col_layout;
            equation_offset = equation_offset,
            block_size = block_size
            )
    end
    return out
end

function inner_sparsity_ct(target_impact, source_impact, nentities_source, nentities_target, n_partials, n_eqs, row_layout::ScalarLayout, col_layout::ScalarLayout; block_size = 1, equation_offset = 0)
    F = eltype(target_impact)
    I = target_impact
    J = source_impact
    I, J = expand_block_indices(I, J, nentities_target, n_eqs, row_layout, equation_offset = equation_offset, block_size = block_size)
    J, I = expand_block_indices(J, I, nentities_source, n_partials, col_layout)

    if block_size > 1
        n = max(block_size, n_eqs)*nentities_target
    else
        n = n_eqs*nentities_target
    end
    m = n_partials*nentities_source
    return SparsePattern(I, J, n, m, row_layout, col_layout)
end

function inner_sparsity_ct(target_impact, source_impact, nentities_source, nentities_target, n_partials, n_eqs, row_layout::T, col_layout::T; block_size = 1, equation_offset = 0) where T<:BlockMajorLayout
    I = target_impact
    J = source_impact
    n = nentities_target
    m = nentities_source
    return SparsePattern(I, J, n, m, row_layout, col_layout)
end

function setup_cross_term_storage(ct::CrossTerm, eq_t, eq_s, model_t, model_s, storage_t, storage_s; ad::Bool = true)
    is_symm = has_symmetry(ct)
    # Find all entities x
    active = cross_term_entities(ct, eq_t, model_t)
    N = length(active)

    state_t = convert_to_immutable_storage(storage_t[:state])
    state_t0 = convert_to_immutable_storage(storage_t[:state0])

    state_s = convert_to_immutable_storage(storage_s[:state])
    state_s0 = convert_to_immutable_storage(storage_s[:state0])

    F_t!(out, state, state0, i) = update_cross_term_in_entity!(out, i, state, state0, as_value(state_s), as_value(state_s0), model_t, model_s, ct, eq_t, 1.0)
    F_s!(out, state, state0, i) = update_cross_term_in_entity!(out, i, as_value(state_t), as_value(state_t0), state, state0, model_t, model_s, ct, eq_t, 1.0)

    n = number_of_equations_per_entity(model_t, eq_t)
    e_t = associated_entity(eq_t)
    ne_t = count_active_entities(model_t.domain, e_t)

    if isnothing(eq_s)
        ne_s = ne_t
        e_s = e_t
    else
        @assert number_of_equations_per_entity(model_s, eq_s) == n
        e_s = associated_entity(eq_s)
        ne_s = count_active_entities(model_s.domain, e_s)
    end
    for i in 1:N
        prepare_cross_term_in_entity!(i, state_t, state_t0, state_s, state_s0, model_t, model_s, ct, eq_t, 1.0)
    end
    caches_t = create_equation_caches(model_t, n, N, storage_t, F_t!, ne_t, self_entity = e_t, ad = ad)
    caches_s = create_equation_caches(model_s, n, N, storage_s, F_s!, ne_s, self_entity = e_s, ad = ad)
    # Extra alignment - for off diagonal blocks
    other_align_t = create_extra_alignment(caches_s, allocate = is_symm)
    out = JutulStorage()
    if is_symm
        other_align_s = create_extra_alignment(caches_t)
        active_source = cross_term_entities_source(ct, eq_s, model_s)
        out[:source_entities] = remap_impact(active_source, model_s, e_s)# active_source
        offdiagonal_alignment = (from_target = other_align_s, from_source = other_align_t)
    else
        offdiagonal_alignment = (from_source = other_align_t, )
    end
    out[:N] = N
    out[:helper_mode] = false
    out[:target] = caches_t
    out[:source] = caches_s
    out[:target_entities] = remap_impact(active, model_t, e_t)
    out[:offdiagonal_alignment] = offdiagonal_alignment
    return out
end

function remap_impact(active, model, entity)
    gmap = global_map(model.domain)
    if gmap isa TrivialGlobalMap
        active_mapped = active
    else
        active_mapped = similar(active)
        for (i, v) in enumerate(active)
            active_mapped[i] = index_map(v, gmap, VariableSet(), EquationSet(), entity)
        end
    end
    return active_mapped
end

function create_extra_alignment(cache; allocate = true)
    out = Dict{Symbol, Any}()
    for k in keys(cache)
        if k == :numeric
            continue
        end
        jp = cache[k].jacobian_positions
        if allocate
            next = similar(jp)
        else
            next = jp
        end
        out[k] = next
    end
    return convert_to_immutable_storage(out)
end

function cross_term_entities(ct::CrossTerm, eq::JutulEquation, model)
    return 1:count_active_entities(model.domain, associated_entity(eq))
end

function cross_term_entities_source(ct, eq, model)
    # Impact on source - if symmetry is present. Should either be no entries (for no symmetry)
    # or equal number of entries (for symmetry)
    return cross_term_entities(ct, eq, model)
end

cross_term_entities_source(ct, eq::Nothing, model) = nothing


function update_main_linearized_system_subgroup!(storage, model, model_keys, offsets, lsys; kwarg...)
    for (index, key) in enumerate(model_keys)
        offset = offsets[index]
        m = model.models[key]
        s = storage[key]
        eqs_s = s.equations
        eqs = m.equations
        eqs_views = s.views.equations
        update_linearized_system!(lsys, eqs, eqs_s, eqs_views, m; equation_offset = offset, kwarg...)
    end
    for (index, key) in enumerate(model_keys)
        offset = offsets[index]
        m = model.models[key]
        eq_views = storage[key].views.equations
        ct, ct_s = cross_term_target(model, storage, key, true)
        update_linearized_system_cross_terms!(lsys, eq_views, ct, ct_s, m, key; equation_offset = offset, kwarg...)
    end
end

function source_impact_for_pair(ctp, ct_s, label)
    sgn = 1
    if ctp.target != label
        # We are dealing with the transposed part, reverse the connection
        impact = ct_s.source_entities
        caches_s = ct_s.target
        caches_t = ct_s.source
        eq_label = ctp.source_equation
        pos = ct_s.offdiagonal_alignment.from_target
        if symmetry(ctp.cross_term) == CTSkewSymmetry()
            sgn = -1
        end
    else
        impact = ct_s.target_entities
        caches_s = ct_s.source
        caches_t = ct_s.target
        eq_label = ctp.target_equation
        pos = ct_s.offdiagonal_alignment.from_source
    end
    return (eq_label, impact, caches_s, caches_t, pos, sgn)
end

function update_linearized_system_cross_terms!(lsys, eq_views, crossterms, crossterm_storage, model, label;
        equation_offset = 0,
        r = lsys.r_buffer,
        nzval = lsys.jac_buffer
    )
    for (ctp, ct_s) in zip(crossterms, crossterm_storage)
        ct = ctp.cross_term
        eq_label, impact, _, caches, _, sgn = source_impact_for_pair(ctp, ct_s, label)
        eq = ct_equation(model, eq_label)
        @assert !isnothing(impact)
        nu = number_of_entities(model, eq)
        r_ct = eq_views[eq_label]
        update_linearized_system_cross_term!(nzval, r_ct, model, ct, caches, impact, nu, sgn)
    end
end

function update_offdiagonal_linearized_system_cross_term!(nz, model, ctp, ct_s, label)
    _, _, caches, _, pos, sgn = source_impact_for_pair(ctp, ct_s, label)
    @assert !isnothing(pos)
    for u in keys(caches)
        if u == :numeric
            continue
        end
        fill_crossterm_entries!(nz, model, caches[u], pos[u], sgn)
    end
end

function update_linearized_system_cross_term!(nz, r, model, ct::AdditiveCrossTerm, caches, impact, nu, sgn)
    for k in keys(caches)
        if k == :numeric
            continue
        end
        increment_equation_entries!(nz, r, model, caches[k], impact, nu, sgn)
    end
end

function update_linearized_system_cross_term!(nz::Missing, r::AbstractArray, model, ct::AdditiveCrossTerm, caches::AbstractArray, impact, nu, sgn)
    increment_equation_entries!(r, model, caches, impact, nu, sgn)
end

function increment_equation_entries!(r, model, entries, impact, nu, sgn)
    ne, nu = size(entries)
    for ui in 1:nu
        i = impact[ui]
        for e in 1:ne
            r[e, i] += sgn*entries[e, ui]
        end
    end
end

function increment_equation_entries!(nz, r, model, cache, impact, nu, sgn)
    nu_local, ne, np = ad_dims(cache)
    entries = cache.entries
    # tb = minbatch(model.context)
    # @batch minbatch = tb for i in 1:nu
    for ui in 1:nu_local
        @inbounds i = impact[ui]
        for (jno, j) in enumerate(vrange(cache, ui))
            @inbounds for e in 1:ne
                a = sgn*entries[e, j]
                if jno == 1
                    @inbounds r[e, i] += a.value
                end
                @inbounds for d = 1:np
                    ix = get_jacobian_pos(cache, j, e, d)
                    nz[ix] += a.partials[d]
                end
            end
        end
    end
end

function update_offdiagonal_blocks!(storage, model, targets, sources;
        lsys = storage.LinearizedSystem,
        r = missing,
        nzval = missing
    )
    if !ismissing(lsys)
        models = model.models
        # for (ctp, ct_s) in zip(model.cross_terms, storage.cross_terms)
        for i in eachindex(model.cross_terms)
            ctp = model.cross_terms[i]
            ct_s = storage.cross_terms[i]
            update_offdiagonal_block_pair!(lsys, ctp, ct_s, storage, model, models, targets, sources)
        end
    end
end

function update_offdiagonal_block_pair!(linearized_system, ctp, ct_s, storage, model, models, targets, sources)
    ct = ctp.cross_term
    t = ctp.target
    s = ctp.source
    if t in targets && s in sources
        lsys = get_linearized_system_model_pair(storage, model, s, t, linearized_system)
        update_offdiagonal_linearized_system_cross_term!(lsys.jac_buffer, models[s], ctp, ct_s, t)
    end
    if has_symmetry(ct) && t in sources && s in targets
        lsys = get_linearized_system_model_pair(storage, model, t, s, linearized_system)
        update_offdiagonal_linearized_system_cross_term!(lsys.jac_buffer, models[t], ctp, ct_s, s)
    end
end

function fill_crossterm_entries!(nz, model, cache::GenericAutoDiffCache, positions, sgn)
    nu, ne, np = ad_dims(cache)
    entries = cache.entries
    tb = minbatch(model.context, nu)
    @batch minbatch = tb for i in 1:nu
        for (jno, j) in enumerate(vrange(cache, i))
            @inbounds for e in 1:ne
                a = sgn*entries[e, j]
                @inbounds for d = 1:np
                    pos = get_jacobian_pos(cache, j, e, d, positions)
                    nz[pos] = a.partials[d]
                end
            end
        end
    end
end

sub_number_of_equations(model::MultiModel) = map(number_of_equations, model.models)
sub_number_of_degrees_of_freedom(model::MultiModel) = map(number_of_degrees_of_freedom, model.models)
sub_block_sizes(model::MultiModel) = map(model_block_size, model.models)

function model_block_size(model)
    layout = matrix_layout(model.context)
    if layout isa ScalarLayout
        bz = 1
    else
        u = only(get_primary_variable_ordered_entities(model))
        bz = degrees_of_freedom_per_entity(model, u)
    end
    return bz
end

function model_block_size(model, offset_eq)
    layout = matrix_layout(model.context)
    if layout isa ScalarLayout
        bz = 1
    else
        bz = 0
        for (k, eq) in model.equations
            if eq == offset_eq
                break
            end
            bz += number_of_equations_per_entity(model, eq)
        end
    end
    return bz
end


function crossterm_subsystem(model, lsys, target, source; diag = false)
    # neqs = map(number_of_equations, model.models)
    # ndofs = map(number_of_degrees_of_freedom, model.models)

    model_keys = submodels_symbols(model)
    groups = model.groups

    function get_group(s)
        g = groups[findfirst(isequal(s), model_keys)]
        g_k = model_keys[groups .== g]
        return (g, g_k)
    end

    if isa(lsys, MultiLinearizedSystem)
        source_g, source_keys = get_group(source)
        target_g, target_keys = get_group(target)
        I = target_g
        if diag
            J = target_g
        else
            J = source_g
        end
        if !isnothing(model.context) && represented_as_adjoint(matrix_layout(model.context))
            J, I = I, J
        end
        lsys = lsys[I, J]
    else
        source_keys = target_keys = model_keys
    end
    return (lsys, target_keys, source_keys)
end

function diagonal_crossterm_alignment!(s_target, ct, lsys, model, target, source, eq_label, impact, equation_offset, variable_offset)
    lsys, target_keys, source_keys = crossterm_subsystem(model, lsys, target, source, diag = true)
    target_model = model[target]
    ndofs = sub_number_of_degrees_of_freedom(model)
    neqs = sub_number_of_equations(model)
    # Diagonal part: Into target equation, and with respect to target variables
    row_offset = local_group_offset(target_keys, target, neqs)
    column_offset = local_group_offset(target_keys, target, ndofs)

    equation_offset += get_equation_offset(target_model, eq_label)
    for target_e in get_primary_variable_ordered_entities(target_model)
        align_to_jacobian!(s_target, ct, lsys.jac, target_model, target_e, impact,
            row_offset = row_offset,
            column_offset = column_offset,
            equation_offset = equation_offset,
            variable_offset = variable_offset
        )
        variable_offset += number_of_degrees_of_freedom(target_model, target_e)
    end
end

function offdiagonal_crossterm_alignment!(s_source, ct, lsys, model, target, source, eq_label, impact, offdiag_alignment, equation_offset, variable_offset)
    lsys, target_keys, source_keys = crossterm_subsystem(model, lsys, target, source, diag = false)
    J = lsys.jac
    ndofs = sub_number_of_degrees_of_freedom(model)
    neqs = sub_number_of_equations(model)
    # Diagonal part: Into target equation, and with respect to target variables
    row_offset = local_group_offset(target_keys, target, neqs)
    column_offset = local_group_offset(source_keys, source, ndofs)

    target_model = model[target]
    source_model = model[source]

    equation_offset += get_equation_offset(target_model, eq_label)

    @assert !isnothing(offdiag_alignment)
    nt = number_of_entities(target_model, ct_equation(target_model, eq_label))
    for source_e in get_primary_variable_ordered_entities(source_model)
        neqs_total = 0
        for (k, eq) in source_model.equations
            if associated_entity(eq) == source_e
                neqs_total += number_of_equations_per_entity(source_model, eq)
            end
        end
        align_to_jacobian!(s_source, ct, J, source_model, source_e, impact,
            equation_offset = equation_offset,
            variable_offset = variable_offset,
            row_offset = row_offset,
            column_offset = column_offset,
            positions = offdiag_alignment,
            number_of_entities_target = nt,
            row_layout = matrix_layout(target_model.context),
            col_layout = matrix_layout(source_model.context),
            number_of_equations_for_entity = neqs_total,
            context = model.context
        )
        variable_offset += number_of_degrees_of_freedom(source_model, source_e)
        a = offdiag_alignment[entity_as_symbol(source_e)]
        @assert maximum(a, init = 0) <= length(lsys.jac_buffer)
    end
end

function align_cross_terms_to_linearized_system!(storage, model::MultiModel; equation_offset = 0, variable_offset = 0)
    cross_terms = model.cross_terms
    cross_term_storage = storage[:cross_terms]
    lsys = storage[:LinearizedSystem]
    for (ctp, ct_s) in zip(cross_terms, cross_term_storage)
        ct = ctp.cross_term
        target = ctp.target
        source = ctp.source
        impact_t = ct_s.target_entities
        eq_label = ctp.target_equation
        o_algn_t = ct_s.offdiagonal_alignment.from_source

        # Align diagonal
        diagonal_crossterm_alignment!(ct_s.target, ct, lsys, model, target, source, eq_label, impact_t, equation_offset, variable_offset)
        # Align offdiagonal
        offdiagonal_crossterm_alignment!(ct_s.source, ct, lsys, model, target, source, eq_label, impact_t, o_algn_t, equation_offset, variable_offset)

        # If symmetry, repeat the process with reversed terms
        if has_symmetry(ct)
            eq_label = ctp.source_equation
            impact_s = ct_s.source_entities
            o_algn_s = ct_s.offdiagonal_alignment.from_target
            diagonal_crossterm_alignment!(ct_s.source, ct, lsys, model, source, target, eq_label, impact_s, equation_offset, variable_offset)
            offdiagonal_crossterm_alignment!(ct_s.target, ct, lsys, model, source, target, eq_label, impact_s, o_algn_s, equation_offset, variable_offset)
        end
    end
end


function setup_cross_terms_storage!(storage, model; ad = true)
    cross_terms = model.cross_terms
    models = model.models

    storage_and_model(t) = (storage[t], models[t])
    v = Vector()
    for ct in cross_terms
        term = ct.cross_term
        s_t, m_t = storage_and_model(ct.target)
        s_s, m_s = storage_and_model(ct.source)
        eq_t = ct_equation(m_t, ct.target_equation)
        if isnothing(symmetry(term))
            eq_s = nothing
        else
            eq_s = ct_equation(m_s, ct.source_equation)
        end
        ct_s = setup_cross_term_storage(term, eq_t, eq_s, m_t, m_s, s_t, s_s, ad = ad)
        push!(v, ct_s)
    end
    storage[:cross_terms] = v
end

function ct_equation(model, eq::Symbol)
    return model.equations[eq]
end

function ct_equation(model, eq::Pair)
    return last(model.equations[last(eq)])
end

function cross_term(storage, target::Symbol)
    return storage[:cross_terms][target]
end

function cross_term_mapper(model, storage, f)
    ind = findall(f, model.cross_terms)
    return (model.cross_terms[ind], storage[:cross_terms][ind])
end

# function cross_term_mapper(model, storage, f)
#     ind = map(f, model.cross_terms)
#     return (model.cross_terms[ind], storage[:cross_terms][ind])
# end

has_symmetry(x) = !isnothing(symmetry(x))
has_symmetry(x::CrossTermPair) = has_symmetry(x.cross_term)

function cross_term_pair(model, storage, source, target, include_symmetry = false)
    if include_symmetry
        f = x -> (x.target == target && x.source == source) ||
                (has_symmetry(x) && (x.target == source && x.source == target))
    else
        f = x -> x.target == target && x.source == source
    end
    return cross_term_mapper(model, storage, f)
end

function cross_term_target(model, storage, target, include_symmetry = false)
    if include_symmetry
        f = x -> x.target == target || (has_symmetry(x.cross_term) && x.source == target)
    else
        f = x -> x.target == target
    end
    return cross_term_mapper(model, storage, f)
end

function cross_term_source(model, storage, source, include_symmetry = false)
    if include_symmetry
        f = x -> x.source == source || (has_symmetry(x.cross_term) && x.target == source)
    else
        f = x -> x.source == source
    end
    return cross_term_mapper(model, storage, f)
end

function extra_cross_term_sparsity(model, storage, target, include_symmetry = true)
    # Get sparsity of cross terms so that they can be included in any generic equations
    ct_pairs, ct_storage = cross_term_target(model, storage, target, include_symmetry)
    # TODO: Maybe this should just be Symbol?
    # Old def was
    # sparsity = Dict{Union{Symbol, Pair}, Any}()
    sparsity = Dict{Symbol, Any}()
    for (ct_p, ct_s) in zip(ct_pairs, ct_storage)
        # Loop over all cross terms that impact target and grab the global sparsity
        # so that this can be added when doing sparsity detection for the model itself.
        is_target = ct_p.target == target
        if is_target
            caches = ct_s.target
            impact = ct_s.target_entities
            eq = ct_p.target_equation
        else
            caches = ct_s.source
            impact = ct_s.source_entities
            eq = ct_p.source_equation
            @assert has_symmetry(ct_p.cross_term)
        end
        if eq isa Pair
            eq = last(eq)
        end
        eq::Symbol
        if !haskey(sparsity, eq)
            sparsity[eq] = Dict{Symbol, Any}()
        end
        eq_d = sparsity[eq]
        model_t = model[target]
        gmap = global_map(model_t.domain)
        equation_t = ct_equation(model_t, eq)
        e = associated_entity(equation_t)
        N = number_of_entities(model_t, equation_t)
        for (k, v) in pairs(caches)
            if k == :numeric
                continue
            end
            ind_for_k = collect_indices(v, impact, N, gmap, e)
            # Merge with existing if found, otherwise just set it
            if haskey(eq_d, k)
                old = eq_d[k]
                for i in 1:N
                    for l in ind_for_k[i]
                        push!(old[i], l)
                    end
                    unique!(old[i])
                end
            else
                eq_d[k] = ind_for_k
            end
        end
    end
    return sparsity
end

function collect_indices(c::GenericAutoDiffCache, impact, N, M, e)
    entities = [Vector{Int64}() for i in 1:N]
    n = length(c.vpos)-1
    for i = 1:n
        # I = index_map(impact[i], M, VariableSet(), EquationSet(), e)
        I = impact[i]
        for var in c.variables[vrange(c, i)]
            push!(entities[I], var)
        end
    end
    return entities
end

can_impact_cross_term(force_t, cross_term) = false

function apply_forces_to_cross_terms!(storage, model::MultiModel, dt, forces; time = NaN, targets = submodels_symbols(model), sources = targets)
    for (ctp, ct_s) in zip(model.cross_terms, storage.cross_terms)
        (; cross_term, target, source) = ctp
        force_t = forces[target]
        apply_forces_to_cross_term!(ct_s, model, storage, cross_term, target, source, targets, dt, force_t, time = time)
        if has_symmetry(cross_term)
            force_s = forces[source]
            apply_forces_to_cross_term!(ct_s, model, storage, cross_term, source, target, sources, dt, force_s, time = time)
        end
    end
end

function apply_forces_to_cross_term!(ct_s, model, storage, cross_term, target, source, targets, dt, forces; kwarg...)
    if !isnothing(forces)
        for force in values(forces)
            if isnothing(force) || !can_impact_cross_term(force, cross_term)
                continue
            end
            if target in targets
                apply_force_to_cross_term!(ct_s, cross_term, target, source, model, storage, dt, force; kwarg...)
            end
        end
    end
end

apply_force_to_cross_term!(ct_s, cross_term, target, source, model, storage, dt, force; time = time) = nothing

export subcrossterm_pair
function subcrossterm_pair(ctp::CrossTermPair, new_model::MultiModel, partition)
    (; target, source, target_equation, source_equation, cross_term) = ctp
    m_t = new_model[target]
    m_s = new_model[source]
    new_cross_term = subcrossterm(cross_term, ctp, m_t, m_s, global_map(m_t.domain), global_map(m_s.domain), partition)
    return CrossTermPair(target, source, target_equation, source_equation, new_cross_term)
end
