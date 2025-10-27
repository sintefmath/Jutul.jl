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
    # (model, state; kwarg...) -> obj
    if i isa Symbol
        # This is hack for the case where we want to evaluate the objective for
        # all steps for sparsity detection but the objective is a sum. So we
        # fake it by taking the first step.
        i = 1
    end
    step = packed_steps[i]
    step_info = step.step_info
    return (model, state; forces = step.forces, kwarg...) -> G(model, state, step_info[:dt], step_info, forces)
end

function objective_evaluator_from_model_and_state(G::AbstractGlobalObjective, model, packed_steps, current_step)
    if ismissing(current_step)
        # Same hack as for AbstractSumObjective
        current_step = 1
    end
    state0 = packed_steps.state0
    step_infos = packed_steps.step_infos
    allstates = Any[objective_state_shallow_copy(s, model) for s in packed_steps.states]
    function obj_eval(model, state;
            parameters = missing, # Parameters - if not in state.
            allforces = missing, # Forces for all steps
            forces = packed_steps.forces[current_step], # Forces for current step - not used in global solve
            input_data = packed_steps.input_data # Input data for setting up the model
        )
        if ismissing(allforces)
            allforces = packed_steps.forces
        else
            allforces = [allforces[step_info[:step]] for step_info in step_infos]
        end
        if ismissing(parameters)
            prm_src = state
        else
            prm_src = parameters
        end
        for i in 1:length(packed_steps)
            if i == current_step
                continue
            end
            allstates[i] = objective_state_reference_parameters(allstates[i], prm_src, model)
        end
        allstates[current_step] = state
        return G(model, state0, allstates, step_infos, allforces, input_data)
    end
    return obj_eval
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
