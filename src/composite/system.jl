Base.getindex(s::CompositeSystem, i::Symbol) = s.systems[i]

function setup_storage_equations!(eqs, storage, model::CompositeModel; kwarg...)
    error()
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
        F!(S, sys, submodel)
    end
    return S
end

function generate_submodel(m::CompositeModel, label::Symbol)
    subsys = m.system[label]
    return SimulationModel(m.domain, subsys, formulation = m.formulation,
                                               context = m.context,
                                               plot_mesh = m.context)
end