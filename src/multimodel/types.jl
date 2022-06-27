struct MultiModelCoupling
    target
    source
    intersection
    issym
    crosstype
    function MultiModelCoupling(target, source, intersection; crosstype = InjectiveCrossTerm, issym = true)
        new(target,source,intersection, issym, crosstype)
    end   
end

struct CrossTermMeta
    label::Symbol
    equation::Union{Symbol, Nothing}
end

abstract type CrossTerm end

struct CrossTermPair
    target::CrossTermMeta
    source::CrossTermMeta
    cross_term::CrossTerm
end

Base.transpose(c::CrossTermPair) = CrossTermPair(c.source, c.target, c.cross_term)

struct MultiModel{M, T} <: JutulModel
    models::M
    cross_terms::Vector{CrossTermPair}
    groups::Union{Vector, Nothing}
    context::Union{JutulContext, Nothing}
    number_of_degrees_of_freedom::T
    reduction::Union{Symbol, Nothing}
    function MultiModel(models; cross_terms = Vector{CrossTermPair}(), groups = nothing, context = nothing, reduction = nothing)
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
            if length(groups) > 1
                @assert num_groups == 2
            else
                reduction = nothing
            end
        end
        if isnothing(groups) && !isnothing(context)
            for (i, m) in enumerate(models)
                if matrix_layout(m.context) != matrix_layout(context)
                    error("No groups provided, but the outer context does not match the inner context for model $i")
                end
            end
        end
        new{typeof(models), typeof(ndof)}(models, cross_terms, groups, context, ndof, reduction)
    end
end




abstract type AdditiveCrossTerm <: CrossTerm end

"""
A cross model term where the dependency is injective and the term is additive:
(each addition to a entity in the target only depends one entity from the source,
and is added into that position upon application)
"""

struct InjectiveCrossTerm{I, E, T, S, SC} <: CrossTerm
    impact::I                      # 2 by N - first row is target, second is source
    entities::E                    # tuple - first tuple is target, second is source
    crossterm_target::T            # The cross-term, with AD values taken relative to the targe
    crossterm_source::S            # Same cross-term, AD values taken relative to the source
    crossterm_source_cache::SC     # The cache that holds crossterm_source together with the entries.
    equations_per_entity::Integer  # Number of equations per impact
    npartials_target::Integer      # Number of partials per equation (in target)
    npartials_source::Integer      # (in source)
    target_symbol::Symbol          # Symbol of target model
    source_symbol::Symbol          # Symbol of source model
    function InjectiveCrossTerm(target_eq, target_model, source_model, intersection = nothing; target = nothing, source = nothing)
        context = target_model.context
        target_entity = associated_entity(target_eq)
        if isnothing(intersection)
            intersection = get_model_intersection(target_entity, target_model, source_model, target, source)
        end
        target_impact, source_impact, target_entity, source_entity = intersection
        @assert !isnothing(target_impact) "Cannot declare cross term when there is no overlap between domains."
        target_impact::AbstractVector
        source_impact::AbstractVector
        noverlap = length(target_impact)
        @assert noverlap == length(source_impact) "Injective source must have one to one mapping between impact and source."
        # Infer Unit from target_eq
        equations_per_entity = number_of_equations_per_entity(target_eq)

        npartials_target = number_of_partials_per_entity(target_model, target_entity)
        npartials_source = number_of_partials_per_entity(source_model, source_entity)

        target_tag = get_entity_tag(target, target_entity)
        c_term_target = allocate_array_ad(equations_per_entity, noverlap, context = context, npartials = npartials_target, tag = target_tag)
        c_term_source_c = CompactAutoDiffCache(equations_per_entity, noverlap, npartials_source, context = context, tag = source, entity = source_entity)
        c_term_source = c_term_source_c.entries

        # Units and overlap - target, then source
        entities = (target = target_entity, source = source_entity)
        overlap = (target = target_impact, source = source_impact)
        new{typeof(overlap), typeof(entities), typeof(c_term_target), typeof(c_term_source), typeof(c_term_source_c)}(overlap, entities, c_term_target, c_term_source, c_term_source_c, equations_per_entity, npartials_target, npartials_source, target, source)
    end
end

function setup_cross_term(target_eq::JutulEquation, target_model, source_model, target, source, intersection, type::Type{InjectiveCrossTerm}; transpose = false)
    if(transpose)
        intersection = transpose_intersection(intersection)
    end
    ct = InjectiveCrossTerm(target_eq, target_model, source_model, intersection; target=target, source=source)
    return ct
end

abstract type CrossTermSymmetry end

struct CTSkewSymmetry <: CrossTermSymmetry end

symmetry(::CrossTerm) = nothing
