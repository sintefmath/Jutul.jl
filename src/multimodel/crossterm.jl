# function align_to_jacobian!(ct::InjectiveCrossTerm, lsys, target::JutulModel, source::JutulModel; equation_offset = 0, variable_offset = 0)
#     cs = ct.crossterm_source_cache
#     jac = lsys.jac
#     impact_target = ct.impact[1]
#     impact_source = ct.impact[2]
#     pentities = get_primary_variable_ordered_entities(source)
#     nu_t = count_active_entities(target.domain, ct.entities.target)
#     for u in pentities
#         nu_s = count_active_entities(source.domain, u)
#         sc = source.context
#         injective_alignment!(cs, nothing, jac, u, sc,
#                                                 target_index = impact_target,
#                                                 source_index = impact_source,
#                                                 layout = lsys.matrix_layout,
#                                                 target_offset = equation_offset,
#                                                 source_offset = variable_offset,
#                                                 number_of_entities_source = nu_s,
#                                                 number_of_entities_target = nu_t)
#         variable_offset += number_of_degrees_of_freedom(source, u)
#     end
# end

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
        n_eqs = number_of_equations_per_entity(model, eq)
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
    # Find all entities x
    active = cross_term_entities(ct, eq_t, model_t)
    active_source = cross_term_entities_source(ct, eq_s, model_s)

    N = length(active)

    state_t = convert_to_immutable_storage(storage_t[:state])
    state_t0 = convert_to_immutable_storage(storage_t[:state0])

    state_s = convert_to_immutable_storage(storage_s[:state])
    state_s0 = convert_to_immutable_storage(storage_s[:state0])

    F_t!(out, state, state0, i) = update_cross_term_in_entity!(out, i, state, state0, as_value(state_s), as_value(state_s0), model_t, model_s, ct, eq_t, 1.0)
    F_s!(out, state, state0, i) = update_cross_term_in_entity!(out, i, as_value(state_t), as_value(state_t0), state, state0, model_t, model_s, ct, eq_t, 1.0)

    caches_t = create_equation_caches(model_t, eq_t, storage_t, F_t!, N)
    if isnothing(eq_s)
        # Note: Sending same equation
        eq_other = eq_t
    else
        eq_other = eq_s
    end
    caches_s = create_equation_caches(model_s, eq_other, storage_s, F_s!, N)
    # Extra alignment - for off diagonal blocks
    other_align_t = create_extra_alignment(caches_t)
    if has_symmetry(ct)
        other_align_s = create_extra_alignment(caches_s)
    else
        other_align_s = nothing
    end

    return (
            target = caches_t, source = caches_s,
            target_entities = active, source_entities = active_source,
            offdiagonal_alignment = (from_target = other_align_t, from_source = other_align_s)
            )
end

function create_extra_alignment(cache)
    out = Dict{Symbol, Any}()
    for k in keys(cache)
        out[k] = similar(cache[k].jacobian_positions)
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
