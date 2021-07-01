

function align_to_jacobian!(ct::InjectiveCrossTerm, jac, target::TervModel, source::TervModel; equation_offset = 0, variable_offset = 0)
    cs = ct.crossterm_source_cache

    layout = matrix_layout(source.context)

    impact_target = ct.impact[1]
    impact_source = ct.impact[2]
    punits = get_primary_variable_ordered_units(source)
    nu_t = count_units(target.domain, ct.units.target)
    for u in punits
        nu_s = count_units(source.domain, u)
        injective_alignment!(cs, jac, u, layout,
                                                target_index = impact_target,
                                                source_index = impact_source,
                                                target_offset = equation_offset,
                                                source_offset = variable_offset,
                                                number_of_units_source = nu_s,
                                                number_of_units_target = nu_t)
        variable_offset += number_of_degrees_of_freedom(source, u)
    end
end

function apply_cross_term!(eq, ct, model_t, model_s, arg...)
    ix = ct.impact.target
    d = get_diagonal_entries(eq)
    # TODO: Why is this allocating?
    d[:, ix] += ct.crossterm_target
end

function update_linearized_system_crossterm!(nz, model_t, model_s, ct::InjectiveCrossTerm)
    fill_equation_entries!(nz, nothing, model_s, ct.crossterm_source_cache)
end

function declare_pattern(target_model, source_model, x::InjectiveCrossTerm, unit)
    source_unit = x.units.source
    if unit == source_unit
        target_impact = x.impact.target
        source_impact = x.impact.source

        out = (target_impact, source_impact)
    else
        out = nothing
    end
    return out
end

function declare_sparsity(target_model, source_model, x::CrossTerm, unit, layout::EquationMajorLayout)
    primitive = declare_pattern(target_model, source_model, x, unit)
    if isnothing(primitive)
        out = nothing
    else
        target_impact = primitive[1]
        source_impact = primitive[2]
        source_unit = x.units.source
        target_unit = x.units.target
        nunits_source = count_units(source_model.domain, source_unit)
        nunits_target = count_units(target_model.domain, target_unit)

        n_partials = x.npartials_source
        n_eqs = x.equations_per_unit
        I = []
        J = []
        for eqno in 1:n_eqs
            for derno in 1:n_partials
                push!(I, target_impact .+ (eqno-1)*nunits_target)
                push!(J, source_impact .+ (derno-1)*nunits_source)
            end
        end
        I = vcat(I...)
        J = vcat(J...)

        n = n_eqs*nunits_target
        m = n_partials*nunits_source
        out = (I, J, n, m)
        @assert maximum(I) <= n "I index exceeded declared row count $n (largest value: $(maximum(I)))"
        @assert maximum(J) <= m "J index exceeded declared column count $m (largest value: $(maximum(J)))"

        @assert minimum(I) >= 1 "I index was lower than 1"
        @assert minimum(J) >= 1 "J index was lower than 1"
    end
    return out
end
