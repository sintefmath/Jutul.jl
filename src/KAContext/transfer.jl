# Transfer utilities for moving a Simulator and its storage to device.
#
# The idea is:
#   1. Set up the model and simulator on the CPU (DefaultContext) so that
#      sparsity detection and equation alignment happen normally.
#   2. Call `transfer_to_device(sim, backend)` to create a new Simulator
#      whose storage arrays (state, state0, equations caches, linearized
#      system) live on the KA-backed device.

"""
    transfer_to_device(sim::Simulator, backend)

Transfer a fully-set-up `Simulator` to device memory. Returns a new
`Simulator` whose model context is a `KernelAbstractionsContext` and whose
storage arrays (state variables, equation caches, linearized system residual
and Jacobian values) are on the target `backend`.

Sparsity patterns, equation alignment and other structural data remain on
the CPU.
"""
function transfer_to_device(sim::Simulator, backend;
        float_t::Type = Float64,
        index_t::Type = Int64
    )
    model = sim.model
    storage = sim.storage

    context = KernelAbstractionsContext(backend, float_t = float_t, index_t = index_t,
                                         matrix_layout = matrix_layout(model.context))

    # Transfer linearized system
    lsys = storage.LinearizedSystem
    lsys_dev = transfer_linearized_system(lsys, backend, float_t, index_t)

    # Transfer state and state0
    state_dev = transfer_state(storage.state, backend)
    state0_dev = transfer_state(storage.state0, backend)

    # Build new storage with device arrays
    storage_dev = transfer_storage(storage, lsys_dev, state_dev, state0_dev, backend, model)

    # Create a new model with the KA context
    model_dev = replace_model_context(model, context)

    return Simulator(sim.executor, model_dev, storage_dev)
end

function transfer_state(state, backend)
    if state isa NamedTuple
        pairs_list = []
        for k in keys(state)
            v = state[k]
            push!(pairs_list, k => transfer_array_to_device(v, backend))
        end
        return NamedTuple(pairs_list)
    elseif state isa JutulStorage
        new_s = JutulStorage()
        for (k, v) in pairs(Jutul.data(state))
            new_s[k] = transfer_array_to_device(v, backend)
        end
        return Jutul.convert_to_immutable_storage(new_s)
    elseif state isa AbstractDict
        new_s = copy(state)
        for (k, v) in state
            new_s[k] = transfer_array_to_device(v, backend)
        end
        return new_s
    else
        return state
    end
end

function transfer_array_to_device(v::AbstractArray, backend)
    return to_device(backend, Array(v))
end

function transfer_array_to_device(v, backend)
    return v
end

function transfer_linearized_system(lsys::LinearizedSystem, backend, float_t, index_t)
    jac = lsys.jac
    if jac isa StaticSparsityMatrixCSR
        jac_dev = transfer_csr_to_device(jac, backend)
    else
        # For CSC or other sparse formats, transfer nzval to device
        jac_dev = jac  # Fallback: keep on CPU
    end
    nzval_dev = nonzeros(jac_dev)

    # Check if r/dx and their buffers are the same array (aliased).
    # For block_size == 1 systems, they are. We must preserve aliasing.
    r_same = lsys.r === lsys.r_buffer
    dx_same = lsys.dx === lsys.dx_buffer

    r_buf_dev = to_device(backend, Array(lsys.r_buffer))
    r_dev = r_same ? r_buf_dev : to_device(backend, Array(lsys.r))

    dx_buf_dev = to_device(backend, Array(lsys.dx_buffer))
    dx_dev = dx_same ? dx_buf_dev : to_device(backend, Array(lsys.dx))

    return LinearizedSystem(jac_dev, r_dev, dx_dev, nzval_dev, r_buf_dev, dx_buf_dev, lsys.matrix_layout)
end

function transfer_storage(storage, lsys_dev, state_dev, state0_dev, backend, model)
    new_storage = JutulStorage()
    # Copy over all existing storage entries
    for (k, v) in pairs(Jutul.data(storage))
        new_storage[k] = v
    end
    # Override with device versions
    new_storage[:LinearizedSystem] = lsys_dev
    new_storage[:state] = state_dev
    new_storage[:state0] = state0_dev

    # Transfer primary_variables references to point to new state
    if haskey(storage, :primary_variables)
        pv = storage.primary_variables
        if pv isa NamedTuple
            new_pv_pairs = []
            for k in keys(pv)
                if haskey(state_dev, k)
                    push!(new_pv_pairs, k => state_dev[k])
                else
                    push!(new_pv_pairs, k => pv[k])
                end
            end
            new_storage[:primary_variables] = NamedTuple(new_pv_pairs)
        else
            new_pv = copy(pv)
            for k in keys(pv)
                if haskey(state_dev, k)
                    new_pv[k] = state_dev[k]
                end
            end
            new_storage[:primary_variables] = new_pv
        end
    end

    # Transfer equation caches to device
    if haskey(storage, :equations)
        new_storage[:equations] = transfer_equation_storage(storage.equations, backend)
    end

    # Transfer parameters to device (parameters are also in state, but need
    # to keep the separate parameters storage in sync)
    if haskey(storage, :parameters)
        new_storage[:parameters] = transfer_state(storage.parameters, backend)
    end

    # Update views - rebuild against device arrays
    new_storage[:views] = Jutul.setup_equations_and_primary_variable_views(new_storage, model, lsys_dev.r_buffer, lsys_dev.dx_buffer)

    return Jutul.convert_to_immutable_storage(new_storage)
end

function transfer_equation_storage(eq_storage, backend)
    if eq_storage isa NamedTuple
        pairs_list = []
        for k in keys(eq_storage)
            push!(pairs_list, k => transfer_eq_cache(eq_storage[k], backend))
        end
        return NamedTuple(pairs_list)
    elseif eq_storage isa AbstractDict
        new_eqs = copy(eq_storage)
        for (k, v) in eq_storage
            new_eqs[k] = transfer_eq_cache(v, backend)
        end
        return new_eqs
    else
        return eq_storage
    end
end

function transfer_eq_cache(cache::CompactAutoDiffCache, backend)
    entries_dev = to_device(backend, Array(cache.entries))
    jpos_dev = to_device(backend, Array(cache.jacobian_positions))
    return CompactAutoDiffCache_device(cache, entries_dev, jpos_dev)
end

function transfer_eq_cache(cache::GenericAutoDiffCache, backend)
    entries_dev = to_device(backend, Array(cache.entries))
    jpos_dev = to_device(backend, Array(cache.jacobian_positions))
    vpos_dev = to_device(backend, Array(cache.vpos))
    vars_dev = to_device(backend, Array(cache.variables))
    dpos = cache.diagonal_positions
    dpos_dev = isnothing(dpos) ? nothing : to_device(backend, Array(dpos))
    return GenericAutoDiffCache_device(cache, entries_dev, jpos_dev, vpos_dev, vars_dev, dpos_dev)
end

function transfer_eq_cache(cache::NamedTuple, backend)
    pairs_list = []
    for k in keys(cache)
        push!(pairs_list, k => transfer_eq_cache(cache[k], backend))
    end
    return NamedTuple(pairs_list)
end

function transfer_eq_cache(cache, backend)
    # Unknown cache type - keep as is
    return cache
end

function replace_model_context(model::SimulationModel, new_context)
    D = typeof(model.domain)
    S = typeof(model.system)
    F = typeof(model.formulation)
    C = typeof(new_context)
    return SimulationModel{D,S,F,C}(
        model.domain,
        model.system,
        new_context,
        model.formulation,
        model.data_domain,
        model.primary_variables,
        model.secondary_variables,
        model.parameters,
        model.equations,
        model.output_variables,
        model.extra,
        model.optimization_level
    )
end
