abstract type CrossTerm end

struct CrossTermPair
    target::Symbol
    source::Symbol
    equation::Symbol
    cross_term::CrossTerm
end

Base.transpose(c::CrossTermPair) = CrossTermPair(c.source, c.target, c.equation, c.cross_term,)

struct MultiModel{T} <: JutulModel
    models::NamedTuple
    cross_terms::Vector{CrossTermPair}
    groups::Union{Vector, Nothing}
    context::Union{JutulContext, Nothing}
    reduction::Union{Symbol, Nothing}
    function MultiModel(models; cross_terms = Vector{CrossTermPair}(), groups = nothing, context = nothing, reduction = nothing, specialize = false)
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
        if reduction == :schur_apply
            if length(groups) > 1
                # @assert num_groups == 2
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
        if specialize
            T = typeof(models)
        else
            T = nothing
        end
        new{T}(models, cross_terms, groups, context, reduction)
    end
end
multi_model_is_specialized(m::MultiModel) = true
multi_model_is_specialized(m::MultiModel{nothing}) = false

submodels(m::MultiModel{nothing}) = m.models
submodels(m::MultiModel{T}) where T = m.models::T

Base.getindex(m::MultiModel, i::Symbol) = submodels(m)[i]

abstract type AdditiveCrossTerm <: CrossTerm end

abstract type CrossTermSymmetry end

struct CTSkewSymmetry <: CrossTermSymmetry end

symmetry(::CrossTerm) = nothing
