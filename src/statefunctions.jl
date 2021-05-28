# TervStateFunction
function update_secondary_variables!(storage, model)
    state = storage.state
    parameters = storage.parameters
    update_secondary_variables!(state, parameters, model)
end

function update_secondary_variables!(state, parameters, model)
    for (symbol, var) in model.secondary_variables
        update_as_secondary!(state[symbol], var, model, state, parameters)
    end
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
    sf = OrderedDict()
    select_primary_variables!(sf, domain, system, formulation)
    return sf
end

function select_primary_variables!(sf, domain, system, formulation)
    select_primary_variables!(sf, system)
end

function minimum_output_variables(domain, system, formulation, primary_variables, secondary_variables)
    minimum_output_variables(system, primary_variables)
end

function minimum_output_variables(system, primary_variables)
    # Default: Output all primary variables
    [i for i in keys(primary_variables)]
end

function map_level(primary_variables, secondary_variables, output_level)
    pkeys = [i for i in keys(primary_variables)]
    skeys = [i for i in keys(secondary_variables)]
    if output_level == :All
        out = vcat(pkeys, skeys)
    elseif output_level == :PrimaryVariables
        out = pkeys
    elseif output_level == :SecondaryVariables
        out = skeys
    else
        out = [output_level]
    end
end

function select_output_variables(domain, system, formulation, primary_variables, secondary_variables, output_level)
    outputs = minimum_output_variables(domain, system, formulation, primary_variables, secondary_variables)
    if !isnothing(output_level)
        if isa(output_level, Symbol)
            output_level  = [output_level]
        end
        for levels in output_level
            mapped = map_level(primary_variables, secondary_variables, levels)
            outputs = vcat(outputs, mapped)
        end
    end
    return outputs
end