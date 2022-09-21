
function get_submodel_offsets(model::MultiModel, group = nothing; is_equation = true)
    if is_equation
        f = number_of_equations
    else
        f = number_of_degrees_of_freedom
    end
    dof = values(map(f, model.models))
    if !isnothing(group)
        dof = dof[model.groups .== group]
    end
    n = length(dof)
    out = zeros(Int64, n)
    tot = 0
    for i in 1:n
        out[i] = tot
        tot += dof[i]
    end
    return out
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
function setup_cross_term(cross_term::CrossTerm; target::Symbol, source::Symbol, equation::Symbol)
    @assert target != source
    return CrossTermPair(target, source, equation, cross_term)
end


function add_cross_term!(v::AbstractVector, ctm::CrossTermPair)
    push!(v, ctm)
    return v
end

function add_cross_term!(model::MultiModel, ctm::CrossTermPair)
    @assert haskey(model.models, ctm.target)
    @assert haskey(model.models, ctm.source)
    @assert haskey(model.models[ctm.target].equations, ctm.equation)
    if has_symmetry(ctm.cross_term)
        @assert haskey(model.models[ctm.source].equations, ctm.equation)
    end
    add_cross_term!(model.cross_terms, ctm)
end

function add_cross_term!(model, cross_term; kwarg...)
    add_cross_term!(model.cross_terms, setup_cross_term(cross_term; kwarg...))
end

