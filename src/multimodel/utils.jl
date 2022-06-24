
function get_submodel_degree_of_freedom_offsets(model::MultiModel, group = nothing)
    dof = values(model.number_of_degrees_of_freedom)
    if !isnothing(group)
        dof = dof[model.groups .== group]
    end
    return cumsum(vcat([0], [i for i in dof]))
end

function get_linearized_system_submodel(storage, model, symbol, lsys = storage.LinearizedSystem)
    return get_linearized_system_model_pair(storage, model, symbol, symbol, lsys)
end

function get_linearized_system_model_pair(storage, model, source, target, lsys = storage.LinearizedSystem)
    if has_groups(model)
        i = group_index(model, target)
        j = group_index(model, source)
        lsys = lsys[i, j]
    end
    return lsys
end

function group_index(model, symbol)
    index = model.groups[findfirst(isequal(symbol), keys(model.models))]
    return index::Integer
end

function submodels_symbols(model::MultiModel)
    return keys(model.models)
end

export setup_cross_term, add_cross_term!
function setup_cross_term(target::Symbol, source::Symbol, cross_term::CrossTerm, target_eq::Symbol, source_eq = nothing)
    @assert target != source
    if isnothing(source_eq) && !isnothing(symmetry(cross_term))
        source_eq = target_eq
    end
    return CrossTermPair(target, source, target_eq, source_eq, cross_term)
end

function add_cross_term!(model::MultiModel, ctm::CrossTermPair)
    push!(model.cross_terms, ctm)
    return model
end

function add_cross_term!(model::MultiModel, arg...)
    add_cross_term!(model, setup_cross_term(arg...))
end

