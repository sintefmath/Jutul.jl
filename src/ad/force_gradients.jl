function force_targets(model, variant = :all)
    out = Dict{Symbol, Union{Nothing, Symbol}}()
    for k in keys(setup_forces(model))
        out[k] = variant
    end
    return out
end

function vectorize_forces(forces, model, targets = force_targets(model); T = Float64)
    meta_for_forces = OrderedDict{Symbol, Any}()
    lengths = vectorization_lengths(forces, model, targets)
    config = (
        lengths = lengths,
        offsets = [1],
        meta = meta_for_forces,
        targets = targets
    )
    n = sum(lengths)
    v = Vector{T}(undef, n)
    vectorize_forces!(v, model, config, forces, update_config = true)
    return (v, config)
end

function vectorization_lengths(forces, model, targets = force_targets(model))
    fkeys = keys(forces)
    lengths = zeros(Int, length(fkeys))
    for (i, force_name) in enumerate(fkeys)
        target = targets[force_name]
        if isnothing(target)
            continue
        end
        force = forces[force_name]
        if isnothing(force)
            continue
        end
        lengths[i] = vectorization_length(force, model, force_name, target)
    end
    return lengths
end

function vectorize_forces!(v, model, config, forces; update_config = false)
    (; meta, lengths, offsets, targets) = config
    lpos = 1
    offset = 0
    for (k, force) in pairs(forces)
        target = targets[k]
        if isnothing(target)
            continue
        end
        if isnothing(force)
            continue
        end
        n_f = lengths[lpos]
        v_f = view(v, (offset+1):(offset+n_f))
        m = vectorize_force!(v_f, model, force, k, target)
        # Update global offset here.
        if update_config
            if vectorization_sublength(force, m) == 1
                push!(offsets, offsets[end] + n_f)
            else
                @assert haskey(m, :lengths) "Bad setup for vector?"
                push!(offsets, offsets[end] + sum(m[:lengths]))
            end
            @assert offsets[end] == offsets[end-1] + n_f
            meta[k] = m
        end
        lpos += 1
        offset += n_f
    end
    return offset
end

function vectorization_length(force, model, name, variant)
    return 1
end

function vectorization_length(force::Vector, model, name, variant)
    return sum(
        x -> vectorization_length(x, model, name, variant),
        force
    )
end

function vectorization_sublength(force, meta)
    return 1
end

function vectorization_sublength(force::Vector, meta)
    return length(meta.lengths)
end

function vectorize_force(x::Jutul.JutulForce, model, name, variant; T = Float64)
    n = vectorization_length(x, model, name, variant)
    v = zeros(T, n)
    vectorize_force!(v, model, x, name, variant)
    return v
end

function vectorize_force!(v, model, forces, name, variant)
    error("Not implemented for $name: $(typeof(forces))")
end

function vectorize_force!(v, model, forces::Vector, name, variant)
    offset = 0
    meta_sub = Vector{Any}(undef, length(forces))
    lengths = Vector{Int}(undef, length(forces))
    for (i, force) in enumerate(forces)
        n_i = vectorization_length(force, model, name, variant)
        v_i = view(v, (offset+1):(offset + n_i))
        meta_sub[i] = vectorize_force!(v_i, model, force, name, variant)
        offset += n_i
        lengths[i] = n_i
    end
    return (meta = meta_sub, lengths = lengths)
end

function devectorize_forces(forces, model, X, config; offset = 0)
    new_forces = OrderedDict{Symbol, Any}()
    lengths = config.lengths
    offset = 0
    ix = 1
    for (k, v) in pairs(forces)
        target = config.targets[k]
        if isnothing(target)
            new_forces[k] = copy(forces[k])
            continue
        end
        if isnothing(v)
            continue
        end
        n_i = lengths[ix]
        X_i = view(X, (offset+1):(offset+n_i))
        new_forces[k] = devectorize_force(v, model, X_i, config.meta[k], k, target)
        offset += n_i
        ix += 1
    end
    return setup_forces(model; new_forces...)
end

function devectorize_force(force::Vector, model, X, cfg, name, variant)
    new_force_any = Vector{Any}(undef, length(force))
    offset = 0
    for (i, f) in enumerate(force)
        n_i = vectorization_length(f, model, name, variant)
        X_i = view(X, (offset+1):(offset+n_i))
        new_force_any[i] = devectorize_force(f, model, X_i, cfg.meta[i], name, variant)
        offset += n_i
    end
    # Narrow type def
    new_force = map(identity, new_force_any)
    return new_force
end

function unique_forces_and_mapping(allforces, timesteps; eachstep = false)
    function force_steps(forces, dt)
        return ([forces], [1:length(dt)], 1)
    end

    function force_steps(forces::Vector, dt)
        length(allforces) == length(dt) || error("Mismatch in length of forces $(length(allforces)) and dt ($(length(dt)))")
        force_map = Vector{Int}[]
        unique_forces = []
        for (j, f) in enumerate(forces)
            found = false
            for (i, uf) in enumerate(unique_forces)
                if isequal(f, uf) || hash(f) == hash(uf)
                    push!(force_map[i], j)
                    found = true
                    break
                end
            end
            if !found
                push!(unique_forces, f)
                push!(force_map, [j])
            end
        end
        num_unique_forces = length(unique_forces)
        @assert sum(length, force_map) == length(forces) "Expected all forces to be accounted for ($(sum(length, force_map)) ≠ $(length(forces)))."
        return (unique_forces, force_map, num_unique_forces)
    end
    if eachstep
        if !(allforces isa Vector)
            allforces = [copy(allforces) for _ in timesteps]
        end
        unique_forces = deepcopy(allforces)
        timesteps_to_forces = collect(eachindex(timesteps))
        forces_to_timestep = [[i] for i in timesteps_to_forces]
        num_unique_forces = length(unique_forces)
    else
        unique_forces, forces_to_timestep, num_unique_forces = force_steps(allforces, timesteps)
        timesteps_to_forces = zeros(Int, length(timesteps))
        for (i, m) in enumerate(forces_to_timestep)
            for j in m
                timesteps_to_forces[j] = i
            end
        end
        @assert all(timesteps_to_forces .> 0)
    end

    return (
        forces = unique_forces,
        forces_to_timesteps = forces_to_timestep,
        timesteps_to_forces = timesteps_to_forces,
        num_unique = num_unique_forces
    )
end

function setup_adjoint_forces_storage(model, states, allforces, timesteps, G;
        n_objective = nothing,
        use_sparsity = true,
        eachstep = false,
        targets = force_targets(model),
        forces_map = missing,
        state0 = setup_state(model),
        parameters = setup_parameters(model),
        single_step_sparsity = false,
        sparsity_step_type = :all, # Note: We could be differentiating each step instead of unique forces, override default
        di_sparse = use_sparsity
    )
    if ismissing(forces_map)
        forces_map = unique_forces_and_mapping(allforces, timesteps, eachstep = eachstep)
    end
    storage = Jutul.setup_adjoint_storage_base(
        model, state0, parameters,
        use_sparsity = use_sparsity,
        n_objective = n_objective,
    )

    unique_forces, forces_to_timestep, timesteps_to_forces, = forces_map

    storage[:forces_map_base] = forces_map
    storage[:forces_map] = forces_map
    storage[:forces_config] = []
    storage[:forces_offsets] = Int[1]
    storage[:targets] = targets

    nvar = storage.n_forward
    offsets = storage[:forces_offsets]
    for (i, force) in enumerate(unique_forces)
        X, config = vectorize_forces(force, model, targets)
        push!(storage[:forces_config], config)
        push!(offsets, offsets[end]+length(X))
    end
    X, = get_adjoint_forces_vectors(model, storage, allforces)
    F = get_adjoint_forces_setup_function(storage, model, parameters, state0)
    packed_steps = AdjointPackedResult(states, timesteps, allforces)
    storage[:adjoint] = Jutul.AdjointsDI.setup_adjoint_storage_generic(X, F, packed_steps, G,
        single_step_sparsity = single_step_sparsity,
        sparsity_step_type = sparsity_step_type,
        di_sparse = di_sparse
    )
    return storage
end

function get_adjoint_forces_vectors(model, storage, allforces)
    offsets = storage[:forces_offsets]::Vector{Int}
    fmap = storage[:forces_map]
    if !haskey(storage, :X)
        N = storage[:forces_offsets][end]-1
        storage[:X] = zeros(N)
        storage[:forces_gradient] = zeros(N)
    end
    X = storage[:X]
    dX = storage[:forces_gradient]
    configs = storage[:forces_config]
    for (i, cfg) in enumerate(configs)
        fno = fmap.forces_to_timesteps[i][1]
        if allforces isa AbstractVector
            forces = allforces[fno]
        else
            @assert length(configs) == 1
            forces = allforces
        end
        X_i = view(X, offsets[i]:(offsets[i+1]-1))
        vectorize_forces!(X_i, model, cfg, forces)
    end
    return (X, dX)
end

function get_adjoint_forces_setup_function(storage, model, parameters, state0)
    offsets = storage[:forces_offsets]::Vector{Int}
    configs = storage[:forces_config]
    function F(X, step_info)
        fmap = storage[:forces_map]
        ix = step_info[:step]
        i = fmap.timesteps_to_forces[ix]
        X_i = view(X, offsets[i]:(offsets[i+1]-1))
        f = deepcopy(fmap.forces[i])
        dt = step_info[:dt]
        cfg = configs[i]
        return adjoint_forces_setup_case(X_i, cfg, f, dt, model, parameters, state0)
    end
    return F
end

function adjoint_forces_setup_case(X, config, forces, dt, model, p, s0)
    forces = devectorize_forces(forces, model, X, config)
    return JutulCase(model, [dt], forces, parameters = p, state0 = s0)
end

"""
    solve_adjoint_forces(case::JutulCase, res::SimResult, G)

Solve the adjoint equations for the forces in `case` given the simulation result
`res` and the objective function `G`.
"""
function solve_adjoint_forces(case::JutulCase, res, G; kwarg...)
    return solve_adjoint_forces(
        case.model, res.states, res.reports, G, case.forces;
        parameters = case.parameters,
        state0 = case.state0,
        kwarg...
    )
end

function solve_adjoint_forces(model, states, reports, G, allforces;
        state0 = setup_state(model),
        timesteps = report_timesteps(reports),
        parameters = setup_parameters(model),
        kwarg...
    )
    storage = setup_adjoint_forces_storage(model, states, allforces, timesteps, G;
        state0 = state0,
        parameters = parameters,
        kwarg...
    )
    return solve_adjoint_forces!(storage, model, states, reports, G, allforces;
        state0 = state0,
        timesteps = timesteps,
        parameters = parameters,
        init = false
    )
end

function solve_adjoint_forces!(storage, model, states, reports, G, allforces;
        state0 = setup_state(model),
        parameters = setup_parameters(model),
        init = true,
        extra_out = true,
        kwarg...
    )
    has_substates = haskey(first(states), :substates)
    packed_steps = AdjointPackedResult(states, reports, allforces)
    step_ix = map(x -> x[:step], packed_steps.step_infos)
    (; forces, forces_to_timesteps, timesteps_to_forces, num_unique) = storage[:forces_map_base]
    # Account for ministeps
    if has_substates
        new_forces_to_timesteps = Vector{Vector{Int}}()
        for i in eachindex(forces_to_timesteps)
            next = Int[]
            old = forces_to_timesteps[i]
            for (new_step_no, j) in enumerate(step_ix)
                if insorted(j, old)
                    push!(next, new_step_no)
                end
            end
            push!(new_forces_to_timesteps, next)
        end
        new_timesteps_to_forces = timesteps_to_forces[step_ix]
    else
        new_timesteps_to_forces = timesteps_to_forces
        new_forces_to_timesteps = forces_to_timesteps
    end
    new_timesteps_to_forces::Vector{Int}
    @assert sum(length, new_forces_to_timesteps) == length(packed_steps)
    storage[:forces_map] = (
        forces = forces,
        forces_to_timesteps = new_forces_to_timesteps,
        timesteps_to_forces = new_timesteps_to_forces,
        num_unique = num_unique
    )

    if allforces isa Vector
        allforces = allforces[step_ix]
    else
        allforces = [allforces for _ in step_ix]
    end
    @assert length(allforces) == length(packed_steps)
    packed_steps.forces = allforces

    X, dX = get_adjoint_forces_vectors(model, storage, allforces)
    F = get_adjoint_forces_setup_function(storage, model, parameters, state0)

    Jutul.AdjointsDI.solve_adjoint_generic!(dX, X, F, storage[:adjoint], packed_steps, G, state0 = state0)

    if extra_out
        dforces = map(
            info -> F(dX, info).forces,
            packed_steps.step_infos[map(first, new_forces_to_timesteps)]
        )
        offsets = storage[:forces_offsets]
        dX_i = map(i -> dX[offsets[i]:(offsets[i+1]-1)], 1:(length(offsets)-1))
        out = (dforces, new_timesteps_to_forces, dX_i, storage[:forces_config])
    else
        out = dX
    end
    return out
end

function forces_optimization_config(
        model,
        allforces,
        timesteps,
        targets = force_targets(model);
        verbose = true,
        active = true,
        rel_min = -Inf,
        rel_max = Inf,
        abs_min = -Inf,
        abs_max = Inf,
        use_scaling = true
    )
    force_map = unique_forces_and_mapping(allforces, timesteps)
    unique_forces, forces_to_timestep, timesteps_to_forces, = force_map
    force_configs = []
    configs = []
    offsets = [1]
    for (force_ix, forces) in enumerate(unique_forces)
        X, config = vectorize_forces(forces, model, targets)
        push!(offsets, offsets[force_ix] + length(X))
        opt_config = OrderedDict{Symbol, Any}()
        meta = config.meta
        ix = 1

        function add_names!(loc, X, force::Vector, meta, ix)
            for (f, m) in zip(force, meta)
                ix = add_names!(loc, X, f, m, ix)
            end
            return ix
        end

        function local_config(v, ix, gix = ix)
            return OrderedDict(
                :base_value => v,
                :active => active,
                :use_scaling => use_scaling,
                :abs_min => abs_min,
                :abs_max => abs_max,
                :rel_min => rel_min,
                :rel_max => rel_max,
                :base_scale => nothing,
                :low => nothing,
                :high => nothing,
                :local_index => ix,
                :global_index => gix
            )
        end

        function add_names!(loc, X, force, meta, ix)
            for name in meta.names
                loc[name] = local_config(X[ix], ix, offsets[force_ix] - 1 + ix)
                ix += 1
            end
            return ix
        end
        for (fname, force) in pairs(forces)
            loc = OrderedDict{Symbol, Any}()
            if isnothing(force)
                continue
            end
            # TODO: Fix this mess.
            if force isa Vector
                ix = add_names!(loc, X, force, meta[fname].meta, ix)
            else
                ix = add_names!(loc, X, force, meta[fname], ix)
            end
            opt_config[fname] = loc
        end
        if verbose
            jutul_message("Forces", "Set number $force_ix")
            cfg_keys = keys(local_config(NaN, 1))
            tmp = Matrix{Any}(undef, length(X), length(cfg_keys))
            dix = 1
            row_labels = []
            for (fname, fdict) in opt_config
                for (subname, subdict) in fdict
                    push!(row_labels, "$dix: $fname.$subname")
                    # pretty_table(subdict, title = "$fname.$subname")
                    for (ii, k) in enumerate(cfg_keys)
                        tmp[dix, ii] = subdict[k]
                    end
                    dix += 1
                end
            end
            H = [i for i in cfg_keys]
            pretty_table(tmp, row_labels = row_labels, header = H)
        end
        push!(configs, opt_config)
        push!(force_configs, config)
    end
    return JutulStorage(
        configs = configs,
        forces = unique_forces,
        forces_config = force_configs,
        forces_to_timesteps = forces_to_timestep,
        timesteps_to_forces = timesteps_to_forces,
        forces_map = force_map,
        offsets = offsets
        )
end

function setup_force_optimization(case, G, opt_config; verbose = true)
    (; model, state0, parameters, dt, forces) = case

    objective_history = Float64[]
    output_data = JutulStorage(
        objective_history = objective_history,
        best_obj = Inf,
    )
    X = Float64[]
    x0 = Float64[]
    xmin = Float64[]
    xmax = Float64[]
    indices_in_X = Int[]
    offsets = [1]
    for (force_ix, config) in enumerate(opt_config.configs)
        for (fkey, fconfig) in config
            ctr = 0
            for (k, v) in fconfig
                val = v[:base_value]
                if v[:active]
                    low = max(v[:abs_min], val*v[:rel_min])
                    hi = min(v[:abs_max], val*v[:rel_max]) + 1e-12

                    if val < low
                        @warn "$fkey.$k base_value $val is below lower bound $low, capping to $low."
                        val = low
                    elseif val > hi
                        @warn "$fkey.$k base_value $val is above upper bound $hi, capping to $hi."
                        val = hi
                    end
                    push!(x0, val)
                    push!(indices_in_X, v[:global_index])
                    push!(xmin, low)
                    push!(xmax, hi)
                    ctr += 1
                end
                push!(X, val)
            end
            push!(offsets, offsets[force_ix] + ctr)
        end
    end
    global_it = 0
    cache = Dict()

    function evaluate_forward(x, g = nothing)
        sim = Simulator(model, state0 = state0, parameters = parameters)
        allforces = opt_config.forces

        for (i, v, j) in zip(indices_in_X, x, 1:length(x))
            X[i] = clamp(v, xmin[j], xmax[j])
        end
        for (i, force) in enumerate(allforces)
            start = opt_config.offsets[i]
            stop = opt_config.offsets[i+1]-1
            X_i = X[start:stop]
            fcfg = opt_config.forces_config[i]
            allforces[i] = devectorize_forces(force, model, X_i, fcfg)
        end
        simforces = allforces[opt_config.timesteps_to_forces]
        global_it += 1
        states, reports = simulate(sim, dt, forces = simforces, extra_timing = false, info_level = -1)
        if !haskey(cache, :storage)
            cache[:storage] = setup_adjoint_forces_storage(
                    model,
                    states,
                    forces,
                    dt,
                    G,
                    state0 = state0,
                    parameters = parameters
                )
        end
        force_adj_storage = cache[:storage]
        output_data[:states] = states
        if !isnothing(g)
            dforces, t_to_f, grad_adj = solve_adjoint_forces!(force_adj_storage, model, states, reports, G, simforces,
            state0 = state0, parameters = parameters, forces_map = opt_config[:forces_map])

            grad_adj = vcat(grad_adj...)

            for (i, j) in enumerate(indices_in_X)
                ∂g = grad_adj[j]
                @assert isfinite(∂g) "Non-finite gradient for global $j local $i"
                g[i] = ∂g
            end
            return g
        else
            obj = Jutul.evaluate_objective(G, model, states, dt, simforces)
            push!(objective_history, obj)
            if obj < output_data[:best_obj]
                output_data[:best_obj] = obj
            end
            if verbose
                fmt = x -> @sprintf("%2.4e", x)
                rel = obj/objective_history[1]
                best = output_data[:best_obj]
                n = length(objective_history)
                jutul_message("Obj. #$n", "$(fmt(obj)) (best: $(fmt(best)), relative: $(fmt(rel)))")
            end
            return obj
        end
    end
    f = x -> evaluate_forward(x)
    g! = (g, x) -> evaluate_forward(x, g)
    return (x0, xmin, xmax, f, g!, output_data)
end

