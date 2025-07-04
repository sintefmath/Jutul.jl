function adjoint_wrap_objective(G, model)
    # Scalar objective:
    force_type = Union{AbstractVector, AbstractDict, NamedTuple}
    # model, state, dt, step_no, forces
    is_sum_obj = hasmethod(G, Tuple{JutulModel, JUTUL_OUTPUT_TYPE, AbstractVector, AbstractDict, force_type})
    # model, state0, states, step_infos, forces, input_data
    is_global_obj = hasmethod(G, Tuple{JutulModel, JUTUL_OUTPUT_TYPE, AbstractVector, AbstractVector{AbstractDict}, force_type, Any})

    if is_sum_obj
        obj = WrappedSumObjective(G)
    else
        if is_global_obj
            obj = WrappedGlobalObjective(G)
        else
            error("Objective function must be a sum of scalar objectives or....")
        end
    end
    return obj
end

function adjoint_wrap_objective(obj::AbstractJutulObjective, model)
    return obj
end

function (WSO::WrappedSumObjective)(model, state, dt, step_info, forces)
    return WSO.objective(model, state, dt, step_info, forces)
end
function (WSO::WrappedGlobalObjective)(model, state0, states, step_infos, forces, case = missing)
    return WSO.objective(model, state0, states, step_infos, forces, case)
end
