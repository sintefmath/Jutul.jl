export substate
function substate(state0_global, m, submod::MultiModel)
    state0 = Dict()
    mods = submod.models
    for k in keys(mods)
        state0[k] = build_init(state0_global[k], m.models[k], mods[k])
    end
    return state0
end

function substate(state0_global, m, submod)
    M = global_map(submod.domain)
    if isa(M, Jutul.TrivialGlobalMap)
        out = setup_state(submod, state0_global)
    else
        partition = M.cells
        init = Dict{Symbol, Any}()
        partition_slice(v::AbstractVector) = v[partition]
        partition_slice(v::AbstractMatrix) = v[:, partition]
        partition_slice(v) = v

        for k in keys(state0_global)
            if haskey(m.primary_variables, k)
                var = m.primary_variables[k]
            else
                continue
                # var = m.secondary_variables[k]
            end
            u = Jutul.associated_entity(var)
            if u == Cells()
                init[k] = partition_slice(state0_global[k])
            else
                error("Not supported")
            end
        end
        out = setup_state(submod, init)
    end
    return out
end
