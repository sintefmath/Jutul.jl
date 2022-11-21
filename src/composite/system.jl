Base.getindex(s::CompositeSystem, i::Symbol) = s.systems[i]

function setup_storage_equations!(eqs, storage, model::CompositeModel; kwarg...)
    subsys_keys = keys(model.system.systems)
    models = JutulStorage()
    for k in subsys_keys
        m = generate_submodel(model, k)
        tmp = OrderedDict()
        setup_storage_equations!(tmp, storage, m; kwarg...)
        for (eq_k, eq_v) in tmp
            @assert !haskey(eqs, eq_k)
            eqs[eq_k] = eq_v
        end
        models[k] = (equations = keys(tmp), model = m)
    end
    storage[:models] = models
    return eqs
end

function select_primary_variables!(S, system::CompositeSystem{T}, model) where T
    internal_select_composite!(S, system, model, select_primary_variables!)
    return S
end

function select_secondary_variables!(S, system::CompositeSystem{T}, model) where T
    primary = select_primary_variables!(OrderedDict(), system, model)
    tmp = OrderedDict()
    internal_select_composite!(tmp, system, model, select_secondary_variables!)
    for (k, v) in tmp
        if !haskey(primary, k)
            S[k] = v
        end
    end
    return S
end

function select_parameters!(S, system::CompositeSystem, model)
    primary = select_primary_variables!(OrderedDict(), system, model)
    secondary = select_secondary_variables!(OrderedDict(), system, model)
    vars = merge!(primary, secondary)
    tmp = OrderedDict()
    internal_select_composite!(tmp, system, model, select_parameters!)
    for (k, v) in tmp
        if !haskey(vars, k)
            S[k] = v
        end
    end
    return S
end

function internal_select_composite!(S, system, model, F!)
    for (name, sys) in pairs(system.systems)
        submodel = generate_submodel(model, name)
        tmp = OrderedDict{Symbol, Any}()
        F!(tmp, sys, submodel)
        for (k, v) in tmp
            S[k] = (name, v)
        end
    end
    return S
end

function generate_submodel(m::CompositeModel, label::Symbol)
    subsys = m.system[label]
    model = SimulationModel(m.domain, subsys, formulation = m.formulation,
                                               context = m.context,
                                               plot_mesh = m.context)
    vars = merge(m.primary_variables, m.secondary_variables, m.parameters)
    for (k, v) in vars
        name, var = v
        if name == label
            replace_variables!(model; k => var)
        end
    end
    return model
end

function setup_forces(model::CompositeModel; kwarg...)
    @warn "Not properly implemented"
    model = generate_submodel(model, first(keys(model.system.systems)))
    forces = setup_forces(model; kwarg...)
    # forces = Dict{Symbol, Any}()
    # for (name, sys) in pairs(system.systems)
    #     submodel = generate_submodel(model, name)
    #     f = setup_forces(submodel)
    #     for (k, v) in pairs(f)
    #         forces[k] = v
    #     end
    # end
    return forces
end

function setup_parameters!(model::JutulModel, init)
    prm = Dict{Symbol, Any}()
    for (name, sys) in pairs(system.systems)
        submodel = generate_submodel(model, name)
        setup_parameters!(prm, submodel, init)
    end
    return prm
end
