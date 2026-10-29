function benchmark_assembly(
        sim::Jutul.JutulSimulator;
        n = 100,
        verbose = false,
        warm = true,
        dt = si_unit(:day),
        forces = setup_forces(sim.model),
        include_secondary = false,
        timer = TimerOutput()
    )
    model = Jutul.get_simulator_model(sim)
    storage = sim.storage
    if warm
        update_state_dependents!(
            storage, model, dt,
            forces,
            time = 0.0,
            update_secondary = true
        )
        Jutul.update_linearized_system!(storage, model)
    end
    @timeit timer "assembly" begin
        for _ in 1:n
            @timeit timer "update_state_dependents" update_state_dependents!(
                storage, model, dt,
                forces,
                time = 0.0,
                update_secondary = include_secondary
            )
            @timeit timer "update_linearized_system" Jutul.update_linearized_system!(storage, model)
        end
    end
    return timer
end
