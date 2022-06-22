

function align_to_jacobian!(ct::InjectiveCrossTerm, lsys, target::JutulModel, source::JutulModel; equation_offset = 0, variable_offset = 0)
    cs = ct.crossterm_source_cache
    jac = lsys.jac
    impact_target = ct.impact[1]
    impact_source = ct.impact[2]
    pentities = get_primary_variable_ordered_entities(source)
    nu_t = count_active_entities(target.domain, ct.entities.target)
    for u in pentities
        nu_s = count_active_entities(source.domain, u)
        sc = source.context
        injective_alignment!(cs, nothing, jac, u, sc,
                                                target_index = impact_target,
                                                source_index = impact_source,
                                                layout = lsys.matrix_layout,
                                                target_offset = equation_offset,
                                                source_offset = variable_offset,
                                                number_of_entities_source = nu_s,
                                                number_of_entities_target = nu_t)
        variable_offset += number_of_degrees_of_freedom(source, u)
    end
end

target_impact(ct) = ct.impact.target

function apply_cross_term!(eq_s, eq, ct, model_t, model_s, arg...)
    ix = target_impact(ct)
    d = get_diagonal_entries(eq, eq_s)
    # NOTE: We do not use += here due to sparse oddities with ForwardDiff.
    increment_diagonal_ct!(d, ct.crossterm_target, ix)
end

increment_diagonal_ct!(d, T, ix) = @tullio d[i, ix[j]] += T[i, j]

function update_linearized_system_crossterm!(nz, model_t, model_s, ct::InjectiveCrossTerm)
    fill_equation_entries!(nz, nothing, model_s, ct.crossterm_source_cache)
end

function declare_pattern(target_model, source_model, x::InjectiveCrossTerm, entity)
    source_entity = x.entities.source
    if entity == source_entity
        target_impact = x.impact.target
        source_impact = x.impact.source

        out = (target_impact, source_impact)
    else
        out = nothing
    end
    return out
end

function declare_sparsity(target_model, source_model, x::CrossTerm, entity, layout::EquationMajorLayout)
    primitive = declare_pattern(target_model, source_model, x, entity)
    if isnothing(primitive)
        out = nothing
    else
        target_impact = primitive[1]
        source_impact = primitive[2]
        source_entity = x.entities.source
        target_entity = x.entities.target
        nentities_source = count_active_entities(source_model.domain, source_entity)
        nentities_target = count_active_entities(target_model.domain, target_entity)

        n_partials = x.npartials_source
        n_eqs = x.equations_per_entity
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
        out = SparsePattern(I, J, n, m, layout)
        @assert maximum(I) <= n "I index exceeded declared row count $n (largest value: $(maximum(I)))"
        @assert maximum(J) <= m "J index exceeded declared column count $m (largest value: $(maximum(J)))"

        @assert minimum(I) >= 1 "I index was lower than 1"
        @assert minimum(J) >= 1 "J index was lower than 1"
    end
    return out
end

function declare_sparsity(target_model, source_model, x::CrossTerm, entity, layout::BlockMajorLayout)
    primitive = declare_pattern(target_model, source_model, x, entity)
    if isnothing(primitive)
        out = nothing
    else
        target_impact = primitive[1]
        source_impact = primitive[2]
        source_entity = x.entities.source
        target_entity = x.entities.target
        nentities_source = count_entities(source_model.domain, source_entity)
        nentities_target = count_entities(target_model.domain, target_entity)

        I = target_impact
        J = source_impact

        n = nentities_target
        m = nentities_source
        out = SparsePattern(I, J, n, m, layout)
    end
    return out
end

function setup_cross_term_storage(ct::CrossTerm, eq, model_t, model_s, storage_t, storage_s)
    # Find all entities x
    state_t = storage_t[:state]
    state_t0 = storage_t[:state0]

    state_s = storage_s[:state]
    state_s0 = storage_s[:state0]

    F_t!(out, state, state0, i) = update_cross_term_in_entity!(out, i, state, state0, as_value(state_s), as_value(state_s0), ct, eq, 1.0)
    F_s!(out, state, state0, i) = update_cross_term_in_entity!(out, i, as_value(state_t), as_value(state_t0), state, state0, ct, eq, 1.0)

    caches_t = create_equation_caches(model_t, eq, storage_t, F_t!)
    caches_s = create_equation_caches(model_s, eq, storage_s, F_s!)

    error()
end

