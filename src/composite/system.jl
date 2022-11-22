Base.getindex(s::CompositeSystem, i::Symbol) = s.systems[i]

function select_primary_variables!(S, system::CompositeSystem{T}, model::CompositeModel) where T
    internal_select_composite!(S, system, model, select_primary_variables!)
    return S
end

function select_secondary_variables!(S, system::CompositeSystem{T}, model::CompositeModel) where T
    primary = model.primary_variables
    tmp = OrderedDict()
    internal_select_composite!(tmp, system, model, select_secondary_variables!)
    for (k, v) in tmp
        if !haskey(primary, k)
            S[k] = v
        end
    end
    return S
end

function select_parameters!(S, system::CompositeSystem, model::CompositeModel)
    primary = model.primary_variables
    secondary = model.secondary_variables
    vars = merge(primary, secondary)
    tmp = OrderedDict()
    internal_select_composite!(tmp, system, model, select_parameters!)
    for (k, v) in tmp
        if !haskey(vars, k)
            S[k] = v
        end
    end
    return S
end


function select_equations!(S, system::CompositeSystem{T}, model::CompositeModel) where T
    internal_select_composite!(S, system, model, select_equations!)
    return S
end

function internal_select_composite!(S, system, model, F!)
    for (name, sys) in pairs(system.systems)
        tmp = OrderedDict{Symbol, Any}()
        F!(tmp, sys, submodel(model, name))
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
