export MultiModel

struct MultiModel <: TervModel
    models::NamedTuple
    groups::Vector
    function MultiModel(models; groups = collect(1:length(models)))
        @assert maximum(groups) <= length(models)
        @assert minimum(groups) > 0
        @assert length(groups) == length(models)
        new(models, groups)
    end
end

abstract type CrossModelTerm end

"""
A cross model term where the dependency is injective and the term is additive:
(each addition to a unit in the target only depends one unit from the source, 
and is added into that position upon application)
"""
struct InjectiveCrossModelTerm
    impact # 2 by N - first row is target, second is source
    units  # tuple - first tuple is target, second is source
    crossterm_target # The cross-term, with AD values taken relative to the target
    crossterm_source # Same cross-term, AD values taken relative to the source
    cross_jac_pos    # Positions that map crossterm_source C into the off-diagonal block ∂C/∂S where S are impacted primary variables of source
    function InjectiveCrossModelTerm(target_eq, target_model, source_model)
        context = target_model.context
        target_unit = domain_unit(target_eq)
        target_impact, source_impact, source_unit = get_domain_intersection(target_unit, target_model, source_model)
        noverlap = length(target_impact)
        @assert noverlap == length(source_impact) "Injective source must have one to one mapping between impact and source."
        I = index_type(context)
        # Infer Unit from target_eq
        equations_per_unit = number_of_equations_per_unit(target_eq)
        npartials_target = number_of_partials_per_unit(target_model, target_unit)
        npartials_source = number_of_partials_per_unit(source_model, source_unit)
        
        alloc = (n) -> allocate_array_ad(equations_per_unit, noverlap, context = context, npartials = n)

        c_term_target = alloc(npartials_target)
        c_term_source = alloc(npartials_source)

        jac_pos = zeros(I, equations_per_unit*npartials_source, noverlap)
        # Units and overlap - target, then source
        units = (target_unit, source_unit)
        overlap = hcat(target_impact, source_impact)
        new(overlap, units, c_term_target, c_term_source, jac_pos)
    end
end

function get_domain_intersection(u::TervUnit, target_model::TervModel, source_model::TervModel)
    return get_domain_intersection(u, target_model.domain, source_model.domain)
end

"""
For a given unit in domain target_d, find any indices into that unit that is connected to
any units in source_d. The interface is limited to a single unit-unit impact.
The return value is a tuple of indices and the corresponding unit
"""
function get_domain_intersection(u::TervUnit, target_d::TervDomain, source_d::TervDomain)
    (nothing, nothing, Cells())
end

function number_of_models(model::MultiModel)
    return length(model.models)
end

function get_primary_variable_names(model::MultiModel)

end

function setup_state!(state, model::MultiModel, init_values)
    error("Mutating version of setup_state not supported for multimodel.")
end

function setup_simulation_storage(model::MultiModel; state0 = setup_state(model), parameters = setup_parameters(model))
    storage = Dict()
    for key in keys(model.models)
        m = model.models[key]
        storage[key] = setup_simulation_storage(m,  state0 = state0[key], 
                                                    parameters = parameters[key], 
                                                    setup_linearized_system = false)
    end
    allocate_cross_model_coupling(storage, model)
    allocate_linearized_system!(storage, model)
    return storage
end

function allocate_cross_model_coupling(storage, model::MultiModel)
    crossd = Dict{Tuple{Symbol, Symbol}, Any}()
    models = model.models
    for target in keys(models)
        for source in keys(models)
            if target == source
                continue
            end
            target_model = models[target]
            source_model = models[source]
            d = Dict()
            for (key, eq) in storage[target][:equations]
                ct = InjectiveCrossModelTerm(eq, target_model, source_model)
                if length(ct.impacted_units) == 0
                    # Just insert nothing, so we can easily spot no overlap
                    ct = nothing
                end
                d[key] = ct
            end
            crossd[(target, source)] = d
        end
    end
    storage[:cross_terms] = crossd
end

function allocate_linearized_system!(storage, model::MultiModel)
    groups = model.groups
    models = model.models
    ugroups = unique(groups)
    ng = length(ugroups)
    if ng == 1
        # All Jacobians are grouped together and we assemble as a single linearized system
        I = []
        J = []
        V = []
        mkeys = keys(models)
        nmodels = length(mkeys)
        rowcounts = zeros(Int64, nmodels)
        colcounts = zeros(Int64, nmodels)

        rowoffset = 0
        coloffset = 0
        for (ix, key) in enumerate(mkeys)
            m = models[key]
            s = storage[key]
            i, j, v, n, m = get_sparse_arguments(s, m)
            rowcounts[ix] = n
            colcounts[ix] = m
            push!(I, i .+ rowoffset)
            push!(J, j .+ coloffset)
            push!(V, v)

        end
        # jac = sparse(I, J, V, )
    else
        # We have multiple groups. Store as Matrix of sparse matrices
        jac = Matrix{Any}(ng, ng)
        row_offset = 0
        col_offset = 0
        for g in 1:ng

        end
    end


    @assert false "Needs implementation"
end

function initialize_storage!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, initialize_storage!)
end

function update_equations!(storage, model::MultiModel, arg...)
    # Might need to update this part
    submodels_storage_apply!(storage, model, update_linearized_system!, arg...)
end

function update_linearized_system!(storage, model::MultiModel, arg...)
    submodels_storage_apply!(storage, model, update_linearized_system!, arg...)
end

function setup_state(model::MultiModel, subs...)
    @assert length(subs) == number_of_models(model)
    state = Dict()
    for (i, key) in enumerate(keys(model.models))
        m = model.models[key]
        state[key] = setup_state(m, subs[i])
    end
    return state
end

function setup_parameters(model::MultiModel)
    p = Dict()
    for key in keys(model.models)
        m = model.models[key]
        p[key] = setup_parameters(m)
    end
    return p
end

function convert_state_ad(model::MultiModel, state)
    stateAD = deepcopy(state)
    for key in keys(model.models)
        stateAD[key] = convert_state_ad(model.models[key], state[key])
    end
    return stateAD
end

function update_properties!(storage, model::MultiModel)
    submodels_storage_apply!(storage, model, update_properties!)
end

function update_equations!(storage, model::MultiModel, dt)
    submodels_storage_apply!(storage, model, update_equations!, dt)
end

function apply_forces!(storage, model::MultiModel, dt, forces::Dict)
    for key in keys(model.models)
        apply_forces!(storage[key], model.models[key], dt, forces[key])
    end
end

function submodels_storage_apply!(storage, model, f!, arg...)
    for key in keys(model.models)
        f!(storage[key], model.models[key], arg...)
    end
end
