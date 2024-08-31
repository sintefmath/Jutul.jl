multi_model_is_specialized(m::MultiModel) = true
multi_model_is_specialized(m::MultiModel{JutulStorage{Nothing}}) = false

function submodel_ad_tag(m::MultiModel, tag)
    if m.specialize_ad
        out = tag
    else
        out = nothing
    end
    return out
end

submodels(m::MultiModel) = m.models

Base.getindex(m::MultiModel, i::Symbol) = submodels(m)[i]

abstract type AdditiveCrossTerm <: CrossTerm end

abstract type CrossTermSymmetry end

struct CTSkewSymmetry <: CrossTermSymmetry end

symmetry(::CrossTerm) = nothing
