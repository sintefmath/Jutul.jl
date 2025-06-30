
function Jutul.simulate(dps::DictParametersSampler, x::AbstractVector)
    prm = optimizer_devectorize!(dps.parameters, x, dps.setup.x_setup)
    case = dps.setup_function(prm, missing)
    result = simulate!(dps.simulator, case.dt,
        config = dps.config,
        state0 = case.state0,
        parameters = case.parameters,
        forces = case.forces
    )
    qoi = dps.output_function(case, result)
    if ismissing(dps.objective)
        out = qoi
    else
        obj = Jutul.evaluate_objective(dps.objective, case.model, result.states, case.dt, case.forces)
        out = (output = qoi, objective = obj)
    end
    return out
end

function Jutul.simulate(dps::DictParametersSampler, x::AbstractMatrix; info_level = 1)
    out = []
    for (i, x_i) in enumerate(eachcol(x))
        if info_level > 0
            println("Simulating parameter set $i/$(size(x, 2))")
        end
        F() = Jutul.simulate(dps, x_i)
        res = redirect_stdout(F, devnull)
        push!(out, res)
    end
    return out
end
