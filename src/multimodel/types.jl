struct MultiModel <: TervModel
    models::NamedTuple
    groups::Union{Vector, Nothing}
    context::Union{TervContext, Nothing}
    number_of_degrees_of_freedom
    reduction
    function MultiModel(models; groups = nothing, context = nothing, reduction = nothing)
        if isnothing(groups)
            num_groups = 1
        else
            nm = length(models)
            num_groups = length(unique(groups))
            @assert maximum(groups) <= nm
            @assert minimum(groups) > 0
            @assert length(groups) == nm
            @assert maximum(groups) == num_groups
        end
        if isa(models, AbstractDict)
            models = convert_to_immutable_storage(models)
        end
        ndof = map(number_of_degrees_of_freedom, models)
        if reduction == :schur_apply
            @assert num_groups == 2
        end
        new(models, groups, context, ndof, reduction)
    end
end

abstract type CrossTerm end

"""
A cross model term where the dependency is injective and the term is additive:
(each addition to a unit in the target only depends one unit from the source,
and is added into that position upon application)
"""
struct InjectiveCrossTerm <: CrossTerm
    impact                 # 2 by N - first row is target, second is source
    units                  # tuple - first tuple is target, second is source
    crossterm_target       # The cross-term, with AD values taken relative to the targe
    crossterm_source       # Same cross-term, AD values taken relative to the source
    crossterm_source_cache # The cache that holds crossterm_source together with the entries.
    equations_per_unit     # Number of equations per impact
    npartials_target       # Number of partials per equation (in target)
    npartials_source       # (in source)
    target_symbol
    source_symbol
    function InjectiveCrossTerm(target_eq, target_model, source_model, intersection = nothing; target = nothing, source = nothing)
        context = target_model.context
        target_unit = associated_unit(target_eq)
        if isnothing(intersection)
            intersection = get_model_intersection(target_unit, target_model, source_model, target, source)
        end
        target_impact, source_impact, target_unit, source_unit = intersection
        @assert !isnothing(target_impact) "Cannot declare cross term when there is no overlap between domains."
        target_impact::AbstractVector
        source_impact::AbstractVector
        noverlap = length(target_impact)
        @assert noverlap == length(source_impact) "Injective source must have one to one mapping between impact and source."
        # Infer Unit from target_eq
        equations_per_unit = number_of_equations_per_unit(target_eq)

        npartials_target = number_of_partials_per_unit(target_model, target_unit)
        npartials_source = number_of_partials_per_unit(source_model, source_unit)

        target_tag = get_unit_tag(target, target_unit)
        c_term_target = allocate_array_ad(equations_per_unit, noverlap, context = context, npartials = npartials_target, tag = target_tag)
        c_term_source_c = CompactAutoDiffCache(equations_per_unit, noverlap, npartials_source, context = context, tag = source, unit = source_unit)
        c_term_source = c_term_source_c.entries

        # Units and overlap - target, then source
        units = (target = target_unit, source = source_unit)
        overlap = (target = target_impact, source = source_impact)
        new(overlap, units, c_term_target, c_term_source, c_term_source_c, equations_per_unit, npartials_target, npartials_source, target, source)
    end
end
