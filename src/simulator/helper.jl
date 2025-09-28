export HelperSimulator, model_residual, model_residual!, model_accumulation, model_accumulation!

struct HelperSimulator{E, M, S, T} <: Jutul.JutulSimulator
    executor::E
    model::M
    storage::S
end

"""
    HelperSimulator(model::M, T = Float64; state0 = setup_state(model), executor::E = Jutul.default_executor()) where {M, E}

Construct a helper simulator that can be used to compute the residuals and/or
accumulation terms for a given type T. Useful for coupling Jutul to other
solvers and types of automatic differentiation.
"""
function HelperSimulator(model::M, T = Float64;
        executor::E = Jutul.default_executor(),
        cache = missing,
        n_extra = 0,
        sort_variables = true,
        kwarg...
    ) where {M, E}
    if sort_variables
        Jutul.sort_secondary_variables!(model)
    end
    n = Jutul.number_of_degrees_of_freedom(model)
    ckey = (T, n)
    has_cache = !ismissing(cache)
    if has_cache && haskey(cache, ckey)
        storage = cache[ckey]
        @assert haskey(storage, :r) "Expected storage to have :r key"
        @assert length(storage[:r]) == n "Expected storage to have length $n"
        @assert eltype(storage[:r]) == T "Expected cached storage to have eltype $T"
    else
        storage = JutulStorage()
        Jutul.setup_storage!(storage, model;
            setup_linearized_system = false,
            state0_ad = false,
            state_ad = false,
            T = T,
            kwarg...
        )
        if n_extra == 0
            r = zeros(T, n)
            r_extended = r
        else
            r_extended = zeros(T, n + n_extra)
            r = view(r_extended, 1:n)
        end
        storage[:r_extended] = r_extended
        storage[:r] = r
        setup_helper_equation_storage!(storage, r, model)
        # TODO: Actually use these.
        # storage[:primary_mapper] = Jutul.variable_mapper(model, :primary)
        # storage[:parameter_wrapper] = first(Jutul.variable_mapper(model, :parameters))
        initialize_extra_state_fields!(storage.state, model, T = T)
        setup_equations_and_primary_variable_views!(storage, model, (dx_buffer = missing, r_buffer = r))
        storage = Jutul.specialize_simulator_storage(storage, model, false)
        if has_cache
            cache[ckey] = storage
        end
    end
    S = typeof(storage)
    return HelperSimulator{E, M, S, T}(executor, model, storage)
end

function HelperSimulator(case::JutulCase, arg...; kwarg...)
    return HelperSimulator(case.model, arg...;
        parameters = case.parameters,
        state0 = case.state0,
        kwarg...
    )
end

"""
    model_residual(sim::HelperSimulator, x, y = missing; kwarg...)

Out of place version of `model_residual!`
"""
function model_residual(sim::HelperSimulator{<:Any, <:Any, <:Any, T}, x, arg...; kwarg...) where T
    model = Jutul.get_simulator_model(sim)
    n = Jutul.number_of_degrees_of_freedom(model)
    @assert length(x) == n "Expected state vector to have $n values, was $(length(x))"
    r = similar(x)
    return model_residual!(r, sim, x, arg...; kwarg...)
end

"""
    model_residual!(r, sim, x, x0 = missing, dt = 1.0;
        forces = setup_forces(sim.model),
        include_accumulation = true,
        kwarg...
    )

Fill in the model residual into Vector r.
"""
function model_residual!(r, sim::HelperSimulator, x, x0 = missing, dt = 1.0;
        forces = setup_forces(sim.model),
        update_secondary = true,
        time = 0.0,
        include_accumulation = true, # Include the accumulation term?
        kwarg...
    )
    storage = get_simulator_storage(sim)
    model = get_simulator_model(sim)
    update_before_step!(sim, dt, forces, time = time)
    if !include_accumulation
        x0 = x
    end

    devectorize_variables!(storage.state, model, x)
    if !ismissing(x0)
        devectorize_variables!(storage.state0, model, x0)
        if update_secondary
            update_secondary_variables!(storage, model, true)
        end
    end

    model = get_simulator_model(sim)
    update_state_dependents!(storage, model, dt, forces; update_secondary = update_secondary, kwarg...) # time is important potential kwarg...
    update_linearized_system!(storage, model, sim.executor, r = storage.r, nzval = missing, lsys = missing)
    @. r = storage.r
    return r
end

function model_residual(sim::HelperSimulator;
        dt = 1.0,
        forces = setup_forces(sim.model),
        update_secondary = true,
        time = 0.0,
        kwarg...
    )
    storage = get_simulator_storage(sim)
    model = get_simulator_model(sim)
    update_before_step!(sim, dt, forces, time = time)

    if update_secondary
        update_secondary_variables!(storage, model, true)
    end

    model = get_simulator_model(sim)
    update_state_dependents!(storage, model, dt, forces; update_secondary = update_secondary, kwarg...) # time is important potential kwarg...
    update_linearized_system!(storage, model, sim.executor, r = storage.r, nzval = missing, lsys = missing)
    return storage.r
end

function model_residual!(r, sim::HelperSimulator;
        kwarg...
    )
    r_internal = model_residual(sim; kwarg...)
    @. r = r_internal
    return r
end

function model_residual(state, state0, sim::HelperSimulator;
        dt = 1.0,
        forces = setup_forces(sim.model),
        time = 0.0,
        kwarg...
    )
    if forces isa AbstractVector
        forces = only(forces)
    end
    function dict_pvar_copy(x, m::MultiModel)
        out = JutulStorage()
        for k in submodels_symbols(m)
            out[k] = dict_pvar_copy(x[k], m[k])
        end
        return out
    end
    function dict_pvar_copy(x, m::SimulationModel)
        pvars = get_primary_variables(m)
        svars = get_secondary_variables(m)
        prms = get_parameters(m)
        out = JutulStorage()
        for k in setdiff(keys(x), svars, prms)
            if haskey(x, k)
                out[k] = x[k]
            end
        end
        return out
    end
    storage = get_simulator_storage(sim)
    model = get_simulator_model(sim)
    r = storage.r
    @. r = 0.0

    state0 = dict_pvar_copy(state0, model)
    state = dict_pvar_copy(state, model)
    # Update the internal state/state0 primary variables
    reset_variables!(storage, model, state0, type = :state0)
    update_secondary_variables!(storage, model, true)
    reset_variables!(storage, model, state, type = :state)
    update_secondary_variables!(storage, model, false)
    update_before_step!(sim, dt, forces, time = time)
    # Update equations and residual
    update_extra_state_fields!(storage, model, dt, time)
    update_state_dependents!(storage, model, dt, forces; update_secondary = false, time = time, kwarg...) # time is important potential kwarg...
    update_linearized_system!(storage, model, sim.executor, r = r, nzval = missing, lsys = missing)
    return r
end

function model_residual!(r, state, state0, sim::HelperSimulator;
        kwarg...
    )
    r_internal = model_residual(state, state0, sim; kwarg...)
    @. r = r_internal
    return r
end

function model_accumulation(sim::HelperSimulator, x, arg...; kwarg...)
    model = Jutul.get_simulator_model(sim)
    n = Jutul.number_of_degrees_of_freedom(model)
    @assert length(x) == n
    acc = similar(x)
    model_accumulation!(acc, sim, x, arg...; kwarg...)
end

"""
    model_accumulation!(acc, sim::HelperSimulator, x, dt = 1.0;
        forces = setup_forces(sim.model),
        update_secondary = true,
        kwarg...
    )

Compute the accumulation term into Vector acc.
"""
function model_accumulation!(acc, sim::HelperSimulator, x, dt = 1.0;
        forces = setup_forces(sim.model),
        update_secondary = true,
        kwarg...
    )
    storage = Jutul.get_simulator_storage(sim)
    model = Jutul.get_simulator_model(sim)
    devectorize_variables!(storage.state, model, x)
    if update_secondary
        Jutul.update_secondary_variables!(storage, model)
    end
    model_accumulation_internal!(acc, storage, model)
    return acc
end

function model_accumulation_internal!(acc, storage, model; offset = 0)
    state = storage.state
    is_cm = is_cell_major(matrix_layout(model.context))
    for (k, eq) in model.equations
        N = Jutul.number_of_equations(model, eq)
        m = Jutul.number_of_equations_per_entity(model, eq)
        n = N ÷ m

        loc_indices = (offset+1):(offset+N)
        acc_i = view(acc, loc_indices)
        if is_cm
            # The equations are always in cell major. We grab a residual view
            # that matches even if the residual is not in cell major.
            v = reshape(acc_i, m, n)
        else
            v = reshape(acc_i, n, m)'
        end
        transfer_accumulation!(acc_i, eq, state)
    end
    return offset
end

function setup_helper_equation_storage!(storage, r, model; offset = 0)
    state = storage.state
    is_cm = is_cell_major(matrix_layout(model.context))
    for (k, eq) in model.equations
        N = Jutul.number_of_equations(model, eq)
        m = Jutul.number_of_equations_per_entity(model, eq)
        n = N ÷ m
        loc_indices = (offset+1):(offset+N)
        r_i = view(r, loc_indices)
        if is_cm
            # The equations are always in cell major. We grab a residual view
            # that matches even if the residual is not in cell major.
            v = reshape(r_i, m, n)
        else
            v = reshape(r_i, n, m)'
        end
        @assert size(v) == (m, n)
        storage[:equations][k] = v
        offset += N
    end
    return offset
end

function select_linear_solver(sim::HelperSimulator; kwarg...)
    return select_linear_solver(sim.model; kwarg...)
end
