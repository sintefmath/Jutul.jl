# TervStateFunction
function update_secondary_variables!(storage, model)
    for (sym, sf) in model.secondary_variables
        update_variable_as_secondary!(storage, model, sf, sym)
    end
end

function update_variable_as_secondary!(storage, model, tv::TervVariables, symbol = get_symbol(tv))
    sf_storage = storage.secondary_variables
    self_storage = sf_storage[symbol]
    parameters = storage.parameters
    update_as_secondary!(self_storage, tv, model, sf_storage, parameters)
end

# Initializers
function select_secondary_variables(domain, system, formulation)
    sf = OrderedDict()
    select_secondary_variables!(sf, domain, system, formulation)
    return sf
end

function select_secondary_variables!(sf, domain, system, formulation)
    select_secondary_variables!(sf, system)
end

function select_secondary_variables!(sf, system)

end

## Definition
function select_primary_variables(domain, system, formulation)
    return select_primary_variables(system)
end

"""
List of symbols that correspond to output variables (= in state0 as numerical values) 
"""
function default_outputs(domain, system, formulation)
    default_outputs(system)
end

function default_outputs(system)
    Vector{Symbol}()
end

function allocate_secondary_variables!(sf_storage, storage, model; tag = nothing)
    for (sym, sf) in model.secondary_variables
        u = associated_unit(sf)
        n = degrees_of_freedom_per_unit(model, u)
        sf_storage[sym] = allocate_secondary_variable(model, sf; npartials = n, tag = tag)
    end
end

function allocate_secondary_variable(model, sf; kwarg...)
    dim = value_dim(model, sf)
    allocate_array_ad(dim...; kwarg...)
end
