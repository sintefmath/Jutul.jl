function benchmark_linear_operators(
        sim::Jutul.JutulSimulator, config = missing;
        n = 100,
        n_update = ceil(Int, n / 20),
        verbose = false,
        warm = true,
        dt = si_unit(:day),
        forces = setup_forces(sim.model),
        timer = TimerOutput(),
        executor = Jutul.default_executor()
    )
    model = Jutul.get_simulator_model(sim)
    storage = sim.storage
    sys = storage.LinearizedSystem
    recorder = Jutul.progress_recorder(sim)
    dummy_timer = TimerOutput()
    context = model.context
    x = missing
    function cycle(loc, x = missing)
        @timeit loc "linear_operator" op = Jutul.linear_operator(sys)
        @timeit loc "residual" r = Jutul.vector_residual(sys)
        if ismissing(x)
            x = similar(r)
        end
        @timeit loc "mul!" mul!(x, op, r)
        return x
    end
    update_state_dependents!(
        storage, model, dt,
        forces,
        time = 0.0,
        update_secondary = true
    )
    # Warm up the linear operator before benchmarking
    if warm
        Jutul.update_linearized_system!(storage, model)
        Jutul.prepare_linear_solve!(sys)
    end

    x = cycle(dummy_timer)
    @timeit timer "linear_operator" begin
        for _ in 1:n
            x = cycle(timer, x)
        end
    end
    if !ismissing(config)
        lsolve = get(config, :linear_solver, missing)
        if lsolve isa GenericKrylov
            prec = lsolve.preconditioner
            r = Jutul.vector_residual(sys)
            update_prec() = Jutul.update_preconditioner!(prec, sys, context, model, storage, recorder, executor)
            apply_prec() = apply!(x, prec, r)
            if warm
                update_prec()
                apply_prec()
            end
            @timeit timer "preconditioner" begin
                if verbose
                    jutul_message("Benchmark", "Updating preconditioner $n_update times")
                end
                for _ in 1:n_update
                    @timeit timer "update_preconditioner" update_prec()
                end
                if verbose
                    jutul_message("Benchmark", "Applying preconditioner $n_update times")
                end
                for _ in 1:n_update
                    @timeit timer "apply!" apply_prec()
                end
            end
        end
    end
    return timer
end
