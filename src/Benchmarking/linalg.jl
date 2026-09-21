function benchmark_linear_operator(sim::Jutul.JutulSimulator;
        n = 100,
        verbose = false,
        warm = true,
        dt = si_unit(:day),
        forces = setup_forces(sim.model),
        timer = TimerOutput()
    )
    model = Jutul.get_simulator_model(sim)
    sys = sim.storage.LinearizedSystem
    dummy_timer = TimerOutput()
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
    update_state_dependents!(sim.storage, model, dt,
        forces,
        time = 0.0,
        update_secondary = true
    )
    # Warm up the linear operator before benchmarking
    Jutul.update_linearized_system!(sim.storage, model)
    Jutul.prepare_linear_solve!(sys)

    x = cycle(dummy_timer)
    @timeit timer "linear_operator" begin
        for i in 1:n
            x = cycle(timer, x)
        end
    end
    return timer
end
