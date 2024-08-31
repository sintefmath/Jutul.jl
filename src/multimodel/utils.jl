
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
    index = model.group_lookup[symbol]
    return index::Int
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

function group_linearized_system_offset(model::MultiModel, target, fn = number_of_degrees_of_freedom)
    models = model.models
    groups = model.groups
    if isnothing(groups)
        groups = ones(Int, length(models))
    end
    skeys = submodels_symbols(model)
    pos = findfirst(isequal(target), skeys)
    g = groups[pos]
    offset = 0
    for (i, k) in enumerate(skeys)
        if k == target
            break
        end
        if groups[i] != g
            continue
        end
        offset += fn(models[k])
    end
    return offset
end

function add_cross_term!(model::MultiModel, ctm::CrossTermPair)
    @assert haskey(model.models, ctm.target)
    @assert haskey(model.models, ctm.source)
    target_label = ctm.target_equation
    if target_label isa Pair
        target_label = target_label[2]
    end
    @assert haskey(model.models[ctm.target].equations, target_label)
    if has_symmetry(ctm.cross_term)
        @assert haskey(model.models[ctm.source].equations, target_label)
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

function initialize_extra_state_fields!(state, mm::MultiModel)
    for (k, model) in pairs(mm.models)
        initialize_extra_state_fields!(state[k], model)
    end
    return state
end