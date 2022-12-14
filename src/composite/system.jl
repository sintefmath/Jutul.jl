Base.getindex(s::CompositeSystem, i::Symbol) = s.systems[i]

function select_primary_variables!(S, v, model::CompositeModel)
    internal_select_composite!(S, v, model, select_primary_variables!)
    return S
end

function select_primary_variables!(S, v::DiscretizedDomain, model::CompositeModel)
    internal_select_composite!(S, v, model, select_primary_variables!)
    return S
end

function composite_select_secondary_variables!(S, v, model::CompositeModel)
    primary = model.primary_variables
    tmp = OrderedDict()
    internal_select_composite!(tmp, v, model, select_secondary_variables!)
    for (k, v) in tmp
        if !haskey(primary, k)
            S[k] = v
        end
    end
    return S
end

select_secondary_variables!(S, v, model::CompositeModel) = composite_select_secondary_variables!(S, v, model)
select_secondary_variables!(S, v::DiscretizedDomain, model::CompositeModel) = composite_select_secondary_variables!(S, v, model)

function composite_select_parameters!(S, v, model::CompositeModel)
    primary = model.primary_variables
    secondary = model.secondary_variables
    vars = merge(primary, secondary)
    tmp = OrderedDict()
    internal_select_composite!(tmp, v, model, select_parameters!)
    for (k, v) in tmp
        if !haskey(vars, k)
            S[k] = v
        end
    end
    return S
end

select_parameters!(S, v, model::CompositeModel) = composite_select_parameters!(S, v, model)
select_parameters!(S, v::DiscretizedDomain, model::CompositeModel) = composite_select_parameters!(S, v, model)

function select_equations!(S, v::DiscretizedDomain, model::CompositeModel)
    internal_select_composite!(S, v, model, select_equations!)
    return S
end

function select_equations!(S, sys::CompositeSystem, model::CompositeModel)
    internal_select_composite!(S, sys, model, select_equations!)
    return S
end

function internal_select_composite!(S, system::CompositeSystem, model, F!)
    for (name, sys) in pairs(system.systems)
        tmp = OrderedDict{Symbol, Any}()
        F!(tmp, sys, submodel(model, name))
        for (k, v) in tmp
            S[k] = (name, v)
        end
    end
    return S
end

function internal_select_composite!(S, something, model, F!)
    for name in keys(model.system.systems)
        tmp = OrderedDict{Symbol, Any}()
        F!(tmp, something, submodel(model, name))
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
    return model
end

function setup_forces(model::CompositeModel; kwarg...)
    force = Dict{Symbol, Any}()
    for k in keys(model.system.systems)
        force[k] = nothing
    end
    for (k, f) in kwarg
        # This should throw if the forces don't match the labels
        submodel(model, k)
        force[k] = f
    end
    return NamedTuple(pairs(force))
end


function apply_forces!(storage, model::CompositeModel, dt, forces; time = NaN)
    equations = model.equations
    equations_storage = storage.equations
    for key in keys(equations)
        eqn = equations[key]
        eq_s = equations_storage[key]
        name, eq = eqn
        k_forces = forces[name]
        if isnothing(k_forces)
            continue
        end
        diag_part = get_diagonal_entries(eq, eq_s)
        for fkey in keys(k_forces)
            force = k_forces[fkey]
            apply_forces_to_equation!(diag_part, storage, submodel(model, name), eq, eq_s, force, time)
        end
    end
end

function setup_parameters!(prm, model::CompositeModel, init)
    # prm = Dict{Symbol, Any}()
    for (name, sys) in pairs(system.systems)
        submodel = submodel(model, name)
        setup_parameters!(prm, submodel, init)
    end
    return prm
end
