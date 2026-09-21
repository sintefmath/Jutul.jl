function benchmark_secondary_variables(sim::JutulSimulator; kwarg...)
    state = evaluation_state(sim)
    model = get_simulator_model(sim)
    return benchmark_secondary_variables(model, state; kwarg...)
end

function benchmark_secondary_variables(
        model::SimulationModel, state;
        warm = true,
        n = 100,
        verbose = false,
        fake_kernel = false,
        use_kernel = model.context isa KernelAbstractionsContext,
        timer = TimerOutput()
    )
    svars = Jutul.get_secondary_variables(model)
    dummy_timer = TimerOutput()
    context = model.context
    function evaluate_variable(k, var, state, model, n, local_timer)
        target = state[k]
        batch_count = length(Jutul.entity_eachindex(target))
        for _ in 1:n
            @timeit local_timer "$k" if use_kernel || fake_kernel
                function update(batch)
                    Jutul.update_secondary_variable!(
                        target, var, model, state, batch
                    )
                    return nothing
                end
                if use_kernel
                    Jutul.KernelExecution.secondary_variable_loop!(state, model, k, context)
                else
                    Jutul.threaded_loop(update, batch_count, context)
                end
                Jutul.synchronize(model.context)
            else
                Jutul.update_secondary_variable!(target, var, model, state)
                Jutul.synchronize(model.context)
            end
        end
        return
    end
    if verbose
        jutul_message("Benchmark", "Starting benchmark of secondary variables...")
    end
    if warm
        for (k, var) in pairs(svars)
            evaluate_variable(k, var, state, model, 1, dummy_timer)  # Warm-up with a single evaluation
        end
    end
    @timeit timer "secondary variables" for (k, var) in pairs(svars)
        if verbose
            jutul_message("Benchmark", "Benchmarking secondary variable $k...")
        end
        t_elapsed = @elapsed evaluate_variable(k, var, state, model, n, timer)
        if verbose
            jutul_message("Benchmark", "Elapsed time for secondary variable $k: $t_elapsed seconds afer $n iterations")
        end
    end

    return timer
end


function benchmark_secondary_variables(
        model::MultiModel, state;
        verbose = false,
        warm = true,
        n = 20
    )
    to = TimerOutput()
    if verbose
        jutul_message("Benchmark", "Starting benchmark of secondary variables...")
    end
    if warm
        if verbose
            jutul_message("Benchmark", "Warming up secondary variables...")
        end
        for k in submodels_symbols(model)
            benchmark_secondary_variables(model[k], state[k]; warm = true, n = 0)
        end
    end

    for k in submodels_symbols(model)
        if verbose
            jutul_message("Benchmark", "Benchmarking secondary variables for submodel $k...")
        end
        submodel = model[k]
        @timeit to "$k" benchmark_secondary_variables(submodel, state[k]; warm = false, n = n, timer = to, verbose = verbose)
    end
    return to
end

# function benchmark_secondary_variables(sim::Simulator{<:Any, <:MultiModel, <:Any}; kwarg...)
#     # state = evaluation_state(sim)
#     model = get_simulator_model(sim)
#     state = JutulStorage()
#     for k in submodels_symbols(model)
#         state[k] = evaluation_state(sim.storage[k])
#     end
#     return benchmark_secondary_variables(model, state; kwarg...)
# end
