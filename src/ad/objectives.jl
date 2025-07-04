function adjoint_wrap_objective(G, model, states, timesteps, forces)
    # Scalar objective:
    # model, state, dt, step_no, forces
    is_sum_obj = applicable(G, model, states[1], timesteps[1], Dict(), forces)
    if is_sum_obj
        obj = WrappedSumObjective(G)
    else
        error("Objective function must be a sum of scalar objectives.")
    end
    return obj
end

function (WSO::WrappedSumObjective)(model, state, dt, step_info, forces)
    return WSO.objective(model, state, dt, step_info, forces)
end
