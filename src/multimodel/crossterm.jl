local_discretization(::CrossTerm, i) = nothing

function update_linearized_system_crossterm!(nz, model_t, model_s, ct::InjectiveCrossTerm)
    fill_equation_entries!(nz, nothing, model_s, ct.crossterm_source_cache)
end


function declare_sparsity(target_model, source_model, eq::JutulEquation, x::CrossTerm, x_storage, entity_indices, target_entity, source_entity, layout)
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
        # n_eqs = x.equations_per_entity
        out = inner_sparsity_ct(target_impact, source_impact, nentities_source, nentities_target, n_partials, n_eqs, layout)
    end
    return out
end

function inner_sparsity_ct(target_impact, source_impact, nentities_source, nentities_target, n_partials, n_eqs, layout::EquationMajorLayout)
    F = eltype(target_impact)
    I = Vector{Vector{F}}()
    J = Vector{Vector{F}}()
    for eqno in 1:n_eqs
        for derno in 1:n_partials
            push!(I, target_impact .+ (eqno-1)*nentities_target)
            push!(J, source_impact .+ (derno-1)*nentities_source)
        end
    end
    I = vcat(I...)
    J = vcat(J...)

    n = n_eqs*nentities_target
    m = n_partials*nentities_source
    return SparsePattern(I, J, n, m, layout)
end

function inner_sparsity_ct(target_impact, source_impact, nentities_source, nentities_target, n_partials, n_eqs, layout::BlockMajorLayout)
    I = target_impact
    J = source_impact
    n = nentities_target
    m = nentities_source
    return SparsePattern(I, J, n, m, layout)
end

function setup_cross_term_storage(ct::CrossTerm, eq_t, eq_s, model_t, model_s, storage_t, storage_s)
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
    ne_t = count_active_entities(model_t.domain, associated_entity(eq_t))

    if !isnothing(eq_s)
        @assert number_of_equations_per_entity(model_s, eq_s) == n
        ne_s = count_active_entities(model_s.domain, associated_entity(eq_s))
    else
        ne_s = ne_t
    end
    for i in 1:N
        prepare_cross_term_in_entity!(i, state_t, state_t0,state_s, state_s0,model_t, model_s, ct, eq_t, 1.0)
    end

    caches_t = create_equation_caches(model_t, n, N, storage_t, F_t!, ne_t)
    caches_s = create_equation_caches(model_s, n, N, storage_s, F_s!, ne_s)
    # Extra alignment - for off diagonal blocks
    other_align_t = create_extra_alignment(caches_s, allocate = is_symm)
    if is_symm
        other_align_s = create_extra_alignment(caches_t)
        active_source = cross_term_entities_source(ct, eq_s, model_s)
        out = (
            N = N, target = caches_t, source = caches_s,
            target_entities = active, source_entities = active_source,
            offdiagonal_alignment = (from_target = other_align_s, from_source = other_align_t)
        )
    else
        out = (
            N = N, target = caches_t, source = caches_s,
            target_entities = active,
            offdiagonal_alignment = (from_source = other_align_t, )
        )
    end
    return out
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

function cross_term_entities(ct, eq, model)
    return 1:count_active_entities(model.domain, associated_entity(eq))
end

function cross_term_entities_source(ct, eq, model)
    # Impact on source - if symmetry is present. Should either be no entries (for no symmetry)
    # or equal number of entries (for symmetry)
    return cross_term_entities(ct, eq, model)
end

cross_term_entities_source(ct, eq::Nothing, model) = nothing


function update_main_linearized_system_subgroup!(storage, model, model_keys, offsets, lsys)
    for (index, key) in enumerate(model_keys)
        offset = offsets[index]
        m = model.models[key]
        s = storage[key]
        eqs_s = s.equations
        eqs = m.equations
        update_linearized_system!(lsys, eqs, eqs_s, m; equation_offset = offset)
    end
    for (index, key) in enumerate(model_keys)
        offset = offsets[index]
        m = model.models[key]
        ct, ct_s = cross_term_target(model, storage, key, true)
        update_linearized_system_cross_terms!(lsys, ct, ct_s, m, key; equation_offset = offset)
    end
end

function source_impact_for_pair(ctp, ct_s, label)
    sgn = 1
    eq_label = ctp.equation
    if ctp.target != label
        # impact = ct_s.target_entities
        impact = ct_s.source_entities
        caches_s = ct_s.target
        caches_t = ct_s.source
        # pos = ct_s.offdiagonal_alignment.from_source
        pos = ct_s.offdiagonal_alignment.from_target
        if symmetry(ctp.cross_term) == CTSkewSymmetry()
            sgn = -1
        end
    else
        # impact = ct_s.source_entities
        impact = ct_s.target_entities
        caches_s = ct_s.source
        caches_t = ct_s.target
        # pos = ct_s.offdiagonal_alignment.from_target
        pos = ct_s.offdiagonal_alignment.from_source
    end
    return (eq_label, impact, caches_s, caches_t, pos, sgn)
end

function update_linearized_system_cross_terms!(lsys, crossterms, crossterm_storage, model, label; equation_offset = 0)
    # @assert !has_groups(model)
    nz = lsys.jac_buffer
    r_buf = lsys.r_buffer
    for (ctp, ct_s) in zip(crossterms, crossterm_storage)
        ct = ctp.cross_term
        eq_label, impact, _, caches, _, sgn = source_impact_for_pair(ctp, ct_s, label)
        eq = model.equations[eq_label]
        @assert !isnothing(impact)
        nu = number_of_entities(model, eq)
        r = local_residual_view(r_buf, model, eq, equation_offset + get_equation_offset(model, eq_label))
        update_linearized_system_cross_term!(nz, r, model, ct, caches, impact, nu, sgn)
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

function update_offdiagonal_blocks!(storage, model, targets, sources)
    linearized_system = storage.LinearizedSystem
    models = model.models
    for (ctp, ct_s) in zip(model.cross_terms, storage.cross_terms)
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
end

function fill_crossterm_entries!(nz, model, cache::GenericAutoDiffCache, positions, sgn)
    nu, ne, np = ad_dims(cache)
    entries = cache.entries
    tb = minbatch(model.context)
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


function update_linearized_system_crossterms!(jac, cross_terms, storage, model::MultiModel, source, target)
    # storage_t, = get_submodel_storage(storage, target)
    model_t, model_s = get_submodels(model, target, source)
    nz = nonzeros(jac)

    for ekey in keys(cross_terms)
        ct = cross_terms[ekey]
        if !isnothing(ct)
            update_linearized_system_crossterm!(nz, model_t, model_s, ct::CrossTerm)
        end
    end
end

function crossterm_subsystem(model, lsys, target, source; diag = false)
    neqs = map(number_of_equations, model.models)
    ndofs = map(number_of_degrees_of_freedom, model.models)

    model_keys = submodel_symbols(model)
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
        lsys = lsys[I, J]
    else
        source_keys = target_keys = model_keys
    end
    row_offset = s -> local_group_offset(target_keys, s, neqs)
    col_offset = s -> local_group_offset(source_keys, s, ndofs)

    target_offsets = (row_offset(target), col_offset(target))
    source_offsets = (row_offset(source), col_offset(source))
    return (lsys, target_offsets, source_offsets)
end

function diagonal_crossterm_alignment!(s_target, ct, lsys, model, target, source, eq_label, impact, equation_offset, variable_offset)
    lsys, target_offset, source_offset = crossterm_subsystem(model, lsys, target, source, diag = true)
    target_model = model[target]
    # Diagonal part: Into target equation, and with respect to target variables
    equation_offset += target_offset[1]
    variable_offset += target_offset[2]

    equation_offset += get_equation_offset(target_model, eq_label)
    for target_e in get_primary_variable_ordered_entities(target_model)
        align_to_jacobian!(s_target, ct, lsys.jac, target_model, target_e, impact, equation_offset = equation_offset, variable_offset = variable_offset)
        variable_offset += number_of_degrees_of_freedom(target_model, target_e)
    end
end

function offdiagonal_crossterm_alignment!(s_source, ct, lsys, model, target, source, eq_label, impact, offdiag_alignment, equation_offset, variable_offset)
    lsys, target_offset, source_offset = crossterm_subsystem(model, lsys, target, source, diag = false)
    equation_offset += target_offset[1]
    variable_offset += source_offset[2]
    target_model = model[target]
    source_model = model[source]

    equation_offset += get_equation_offset(target_model, eq_label)
    @assert !isnothing(offdiag_alignment)
    # @assert keys(s_source), :numeric == keys(offdiag_alignment)
    nt = number_of_entities(target_model, target_model.equations[eq_label])
    for source_e in get_primary_variable_ordered_entities(source_model)
        align_to_jacobian!(s_source, ct, lsys.jac, source_model, source_e, impact, equation_offset = equation_offset,
                                                                                   variable_offset = variable_offset,
                                                                                   positions = offdiag_alignment,
                                                                                   number_of_entities_target = nt,
                                                                                   context = model.context)
        variable_offset += number_of_degrees_of_freedom(source_model, source_e)
        a = offdiag_alignment[Symbol(source_e)]
        if length(a) > 0
            @assert maximum(a) <= length(lsys.jac_buffer)
        end
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
        eq_label = ctp.equation
        o_algn_t = ct_s.offdiagonal_alignment.from_source

        # Align diagonal
        diagonal_crossterm_alignment!(ct_s.target, ct, lsys, model, target, source, eq_label, impact_t, equation_offset, variable_offset)
        # Align offdiagonal
        offdiagonal_crossterm_alignment!(ct_s.source, ct, lsys, model, target, source, eq_label, impact_t, o_algn_t, equation_offset, variable_offset)

        # If symmetry, repeat the process with reversed terms
        if has_symmetry(ct)
            impact_s = ct_s.source_entities
            o_algn_s = ct_s.offdiagonal_alignment.from_target
            diagonal_crossterm_alignment!(ct_s.source, ct, lsys, model, source, target, eq_label, impact_s, equation_offset, variable_offset)
            offdiagonal_crossterm_alignment!(ct_s.target, ct, lsys, model, source, target, eq_label, impact_s, o_algn_s, equation_offset, variable_offset)
        end
    end
end


function setup_cross_terms_storage!(storage, model)
    cross_terms = model.cross_terms
    models = model.models

    storage_and_model(t) = (storage[t], models[t])
    v = Vector()
    for ct in cross_terms
        term = ct.cross_term
        s_t, m_t = storage_and_model(ct.target)
        s_s, m_s = storage_and_model(ct.source)
        eq_t = m_t.equations[ct.equation]
        if isnothing(symmetry(term))
            eq_s = nothing
        else
            eq_s = m_s.equations[ct.equation]
        end

        ct_s = setup_cross_term_storage(term, eq_t, eq_s, m_t, m_s, s_t, s_s)
        push!(v, ct_s)
    end
    storage[:cross_terms] = v
end


function cross_term(storage, target::Symbol)
    return storage[:cross_terms][target]
end

function cross_term_mapper(model, storage, f)
    ind = map(f, model.cross_terms)
    return (model.cross_terms[ind], storage[:cross_terms][ind])
end

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
    function collect_indices(c::GenericAutoDiffCache, impact, N)
        entities = [Vector{Int64}() for i in 1:N]
        n = length(c.vpos)-1
        for i = 1:n
            I = impact[i]
            entities[I] = c.variables[vrange(c, i)]
        end
        return entities
    end
    ct_pairs, ct_storage = cross_term_target(model, storage, target, include_symmetry)
    sparsity = Dict{Symbol, Any}()
    for (ct_p, ct_s) in zip(ct_pairs, ct_storage)
        # Loop over all cross terms that impact target and grab the global sparsity
        # so that this can be added when doing sparsity detection for the model itself.
        is_target = ct_p.target == target
        if is_target
            caches = ct_s.target
            impact = ct_s.target_entities
        else
            caches = ct_s.source
            impact = ct_s.source_entities
            @assert has_symmetry(ct_p.cross_term)
        end
        eq = ct_p.equation
        if !haskey(sparsity, eq)
            sparsity[eq] = Dict{Symbol, Any}()
        end
        eq_d = sparsity[eq]
        model_t = model[target]
        equation_t = model_t.equations[eq]
        N = number_of_entities(model_t, equation_t)
        for (k, v) in pairs(caches)
            if k == :numeric
                continue
            end
            ind_for_k = collect_indices(v, impact, N)
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
        if target in targets
            for force in values(forces)
                if isnothing(force)
                    continue
                end
                apply_force_to_cross_term!(ct_s, cross_term, target, source, model, storage, dt, force; kwarg...)
            end
        end
    end
end

apply_force_to_cross_term!(ct_s, cross_term, target, source, model, storage, dt, force; time = time) = nothing
