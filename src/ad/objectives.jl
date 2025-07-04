function adjoint_wrap_objective(G, model)
    # Scalar objective:
    # model, state, dt, step_no, forces
    is_sum_obj = applicable(G, model, JUTUL_OUTPUT_TYPE(), [1.0], Dict(), [])
    if is_sum_obj
        obj = WrappedSumObjective(G)
    else
        error("Objective function must be a sum of scalar objectives.")
    end
    return obj
end

function adjoint_wrap_objective(obj::AbstractJutulObjective, model)
    return obj
end

function (WSO::WrappedSumObjective)(model, state, dt, step_info, forces)
    return WSO.objective(model, state, dt, step_info, forces)
end
