reservoir_model(model) = model
reservoir_storage(model, storage) = storage
reservoir_storage(model::MultiModel, storage) = storage.Reservoir
reservoir_model(model::MultiModel) = model.models.Reservoir


export setup_reservoir_simulator
function setup_reservoir_simulator(models, initializer, parameters = nothing; method = :cpr, rtol = 0.005, initial_dt = 3600.0*24.0, target_its = 8, offset_its = 1, kwarg...)
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
    lsolve = reservoir_linsolve(mmodel, method, rtol = rtol)
    # day = 3600.0*24.0
    t_base = TimestepSelector(initial_absolute = initial_dt, max = Inf)
    t_its = IterationTimestepSelector(target_its, offset = offset_its)
    cfg = simulator_config(sim, timestep_selectors = [t_base, t_its], linear_solver = lsolve; kwarg...)

    return (sim, cfg)
end

export well_output, well_symbols

function well_output(model::MultiModel, parameters, states, well_symbol, target = BottomHolePressureTarget)
    n = length(states)
    d = zeros(n)

    group = :Facility
    rhoS_o = parameters[well_symbol][:reference_densities]

    target_limit = target(1.0)

    pos = get_well_position(model.models[group].domain, well_symbol)
    well_model = model.models[well_symbol]
    for (i, state) = enumerate(states)
        well_state = state[well_symbol]
        well_state = convert_to_immutable_storage(well_state)
        q_t = state[group][:TotalSurfaceMassRate][pos]
        if q_t == 0
            current_control = DisabledControl()
            d[i] = missing
        else
            if q_t < 0
                current_control = ProducerControl(BottomHolePressureTarget(1.0))
            else
                current_control = InjectorControl(BottomHolePressureTarget(1.0), 1.0)
            end
            rhoS, S = flash_wellstream_at_surface(well_model, well_state, rhoS_o)
            v = well_target_value(q_t, current_control, target_limit, well_model, well_state, rhoS, S)
            d[i] = v
        end
    end
    return d
end

function well_symbols(model::MultiModel)
    models = model.models
    symbols = Vector{Symbol}()
    for (k, m) in pairs(models)
        D = m.domain
        if isa(D, DiscretizedDomain) && isa(D.grid, WellGrid)
            push!(symbols, k)
        end
    end
    return symbols
end
