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

function (WSO::WrappedGlobalObjective)(model, state0, states, step_infos, forces, input_data = missing)
    return WSO.objective(model, state0, states, step_infos, forces, input_data)
end

function objective_evaluator_from_model_and_state(G::AbstractSumObjective, model, packed_steps, i)
    # (model, state) -> obj
    step = packed_steps[i]
    step_info = step.step_info
    return (model, state) -> G(model, state, step_info[:dt], step_info, step.forces)
end

function objective_evaluator_from_model_and_state(G::AbstractGlobalObjective, model, packed_steps, current_step)
    # (model, state) -> obj

    # G(model, state0, allstates, step_infos, allforces, input_data)
    # Create copy of states
    # Make sure that all parameters are references in each state apart from the one we are evaluating

    # Make shallow copies of states
    # Needs to be nested shallow copy if multimodel
    state0 = packed_steps.state0
    step_infos = packed_steps.step_infos
    allforces = packed_steps.forces
    input_data = missing
    allstates = Any[objective_state_shallow_copy(s, model) for s in packed_steps.states]
    function myfunc(model, state)
        # G(model, state0, allstates, step_infos, allforces, input_data)
        # references all parameters (needs to be multimodel aware)
        for i in 1:length(packed_steps)
            if i == current_step
                continue
            end
            allstates[i] = objective_state_reference_parameters(allstates[i], state, model)
        end
        allstates[current_step] = state
        return G(model, state0, allstates, step_infos, allforces, input_data)
    end
    return myfunc
end

function objective_state_reference_parameters(target_state, state, model::SimulationModel)
    for k in keys(model.parameters)
        target_state[k] = state[k]
    end
    return target_state
end

function objective_state_shallow_copy(state, model::SimulationModel)
    target_state = JutulStorage()
    for k in keys(state)
        target_state[k] = state[k]
    end
    return target_state
end
