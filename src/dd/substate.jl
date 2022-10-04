export substate
function substate(state0_global, m::MultiModel, submod::MultiModel, type = :primary)
    state0 = Dict()
    mods = submod.models
    for k in keys(mods)
        state0[k] = substate(state0_global[k], m.models[k], mods[k], type)
    end
    return state0
end

function substate(state0_global, m::SimulationModel, submod::SimulationModel, type = :primary)
    M = global_map(submod.domain)
    if isa(M, Jutul.TrivialGlobalMap)
        out = deepcopy(state0_global)
    else
        if type == :primary
            vars = m.primary_variables
        elseif type == :secondary
            vars = m.secondary_variables
        else
            @assert type == :parameters
            vars = m.parameters
        end
        out = Dict{Symbol, Any}()
        for k in keys(state0_global)
            if haskey(vars, k)
                var = vars[k]
            else
                continue
            end
            e = associated_entity(var)
            p = entity_partition(M, e)
            out[k] = partition_variable_slice(state0_global[k], p)
        end
    end
    return out
end
