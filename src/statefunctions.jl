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

function map_level(primary_variables, secondary_variables, output_level)
    if output_level == :All
        out = vcat(keys(primary_variables), keys(secondary_variables))
    elseif output_level == :Primary
        out = keys(secondary_variables)
    elseif output_level == :Secondary
        out = keys(primary_variables)
    else
        out = [output_level]
    end
end

function select_outputs(domain, system, formulation, primary_variables, secondary_variables, output_level)
    outputs = minimum_outputs(domain, system, formulation)
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