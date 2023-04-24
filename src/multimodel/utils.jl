
function get_submodel_offsets(model::MultiModel, group::Union{Nothing,Integer} = nothing; is_equation = true)
    if is_equation
        f = number_of_equations
    else
        f = number_of_degrees_of_freedom
    end
    dof = values(map(f, model.models))
    bz = values(sub_block_sizes(model))
    if !isnothing(group)
        active = model.groups .== group
        dof = dof[active]
        bz = bz[active]
    end
    n = length(dof)
    out = zeros(Int64, n)
    tot = 0
    for i in 1:n
        out[i] = tot
        tot += dof[i]÷bz[i]
    end
    return out
end

function get_submodel_offsets(storage::StateType, g::Integer)
    return storage.multi_model_maps.offset_map[g]
end

function get_submodel_offsets(storage::StateType)
    return storage.multi_model_maps.offset_map
end

function get_linearized_system_submodel(storage, model, symbol, lsys = storage.LinearizedSystem)
    return get_linearized_system_model_pair(storage, model, symbol, symbol, lsys)
end

function get_linearized_system_model_pair(storage, model, source, target, lsys = storage.LinearizedSystem)
    if has_groups(model)
        i = group_index(model, target)
        j = group_index(model, source)
        if !isnothing(model.context) && represented_as_adjoint(matrix_layout(model.context))
            j, i = i, j
        end
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
function setup_cross_term(cross_term::CrossTerm; target::Symbol, source::Symbol, equation)
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
    @assert haskey(model.models[ctm.target].equations, ctm.target_equation)
    if has_symmetry(ctm.cross_term)
        @assert haskey(model.models[ctm.source].equations, ctm.target_equation)
    end
    add_cross_term!(model.cross_terms, ctm)
end

function add_cross_term!(model, cross_term; kwarg...)
    add_cross_term!(model.cross_terms, setup_cross_term(cross_term; kwarg...))
end

select_linear_solver(model::MultiModel; kwarg...) = select_linear_solver_multimodel(model, first(model.models); kwarg...)
select_linear_solver_multimodel(model::MultiModel, first_model; kwarg...) = select_linear_solver(first_model; kwarg...)

function error_sum_scaled(model::MultiModel, rep)
    err_sum = 0.0
    for (k, v) in rep
        err_sum += error_sum_scaled(model[k], v)
    end
    return err_sum
end
