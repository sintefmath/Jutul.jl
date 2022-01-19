reservoir_model(model) = model
reservoir_storage(model, storage) = storage
reservoir_storage(model::MultiModel, storage) = storage.Reservoir
reservoir_model(model::MultiModel) = model.models.Reservoir


export setup_reservoir_simulator
function setup_reservoir_simulator(models, initializer, parameters = nothing; kwarg...)
    reservoir_model = models[:Reservoir]
    # Convert to multi model
    block_backend = is_cell_major(matrix_layout(reservoir_model.context))
    if block_backend && length(models) > 1
        groups = repeat([2], length(models))
        groups[1] = 1
        red = :schur_apply
        outer_context = DefaultContext()
    else
        outer_context = models[:Reservoir].context
        groups = nothing
        red = nothing
    end
    mmodel = MultiModel(convert_to_immutable_storage(models), groups = groups, 
                                                              context = outer_context,
                                                              reduction = red)
    # Set up simulator itself, containing the initial state
    state0 = setup_state(mmodel, initializer)
    sim = Simulator(mmodel, state0 = state0, parameters = deepcopy(parameters))

    # Config: Linear solver, timestep selection defaults, etc...
    lsolve = reservoir_linsolve(mmodel)
    day = 3600.0*24.0
    t_base = TimestepSelector(initial_absolute = 1*day, max = Inf)
    t_its = IterationTimestepSelector(8)
    cfg = simulator_config(sim, timestep_selectors = [t_base, t_its], linear_solver = lsolve; kwarg...)

    return (sim, cfg)
end
