function force_targets(model, variant = :all)
    out = Dict{Symbol, Union{Nothing, Symbol}}()
    for k in keys(setup_forces(model))
        out[k] = variant
    end
    return out
end

function vectorize_forces(forces, model, targets = force_targets(model); T = Float64)
    meta_for_forces = Dict{Symbol, Any}()
    lengths = vectorization_lengths(forces, model, targets)
    config = (
        lengths = lengths,
        offsets = [1],
        meta = meta_for_forces,
        targets = targets
    )
    n = sum(lengths)
    v = Vector{T}(undef, n)
    vectorize_forces!(v, model, config, forces)
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

function vectorize_forces!(v, model, config, forces)
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
        if vectorization_sublength(force, m) == 1
            push!(offsets, offsets[end] + n_f)
        else
            @assert haskey(m, lengths) "Bad setup for vector?"
            for l in m[:lengths]
                push!(offsets, offsets[end] + l)
            end
        end
        @assert offsets[end] == offsets[end-1] + n_f
        meta[k] = m
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

function devectorize_forces(forces, model, X, config; offset = 0, ad_key = nothing)
    new_forces = OrderedDict{Symbol, Any}()
    lengths = config.lengths
    offset = 0
    ix = 1
    if isnothing(ad_key)
        X_eval = X
    else
        ad_key::Symbol
        offsets = config.offsets
        npartials = maximum(diff(offsets), init = 0)
        if npartials == 0
            X_eval = X
        else
            sample = Jutul.get_ad_entity_scalar(1.0, npartials, 1, tag = ad_key)
            # Initialize acc + forces with that size ForwardDiff.Dual
            T = typeof(sample)
            X_ad = Vector{T}(undef, length(X))
            for fno in 1:(length(offsets)-1)
                k = keys(forces)[fno]
                is_ad = k == ad_key
                local_index = 1
                for j in offsets[fno]:(offsets[fno+1]-1)
                    X_j = X[j]
                    if is_ad
                        X_j = Jutul.get_ad_entity_scalar(X_j, npartials, local_index, tag = ad_key)
                        local_index +=1
                    end
                    X_ad[j] = X_j
                end
            end
            X_eval = X_ad
        end
    end
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
        X_i = view(X_eval, (offset+1):(offset+n_i))
        new_forces[k] = devectorize_force(v, model, X_i, config.meta[k], k, target)
        offset += n_i
        ix += 1
    end
    return setup_forces(model; new_forces...)
end

function devectorize_force(force::Vector, model, X, meta, name, variant)
    # new_force = similar(force)
    new_force_any = Vector{Any}(undef, length(force))
    offset = 0
    for (i, f) in enumerate(force)
        n_i = vectorization_length(f, model, name, variant)
        X_i = view(X, (offset+1):(offset+n_i))
        new_force_any[i] = devectorize_force(f, model, X_i, meta[i], name, variant)
        offset += n_i
    end
    # Narrow type def
    new_force = map(identity, new_force_any)
    return new_force
end

function determine_sparsity_forces(model, forces, X, config;
        parameters = setup_parameters(model),
        extra_sparsity = Dict()
    )
    function fake_state(model)
        init = Dict{Symbol, Any}()
        for (k, v) in Jutul.get_variables(model)
            init[k] = 1.0
        end
        # Also add parameters here.
        state = setup_state(model, init)
        return convert_to_immutable_storage(merge(state, parameters))
    end
    state = fake_state(model)
    storage = (state = state, )
    X_ad = ST.create_advec(X)
    T = eltype(X_ad)
    forces_ad = devectorize_forces(forces, model, X_ad, config)
    sparsity = OrderedDict{Symbol, Any}()
    for (fname, force) in pairs(forces_ad)
        sparsity[fname] = determine_sparsity_force(storage, model, force, T, extra_sparsity = extra_sparsity)
    end

    return sparsity
end

function determine_sparsity_force(storage, model, force_as_stracer, T, offset = 0; extra_sparsity = Dict())
    sparsity = OrderedDict{Symbol, Any}()
    for (eqname, eq) in model.equations
        entity = Jutul.associated_entity(eq)
        nentity = number_of_entities(model, eq)
        neqs = number_of_equations_per_entity(model, eq)
        npartials = Jutul.number_of_partials_per_entity(model, entity)
        acc = zeros(T, neqs, nentity)
        eq_s = missing
        time = NaN
        Jutul.apply_forces_to_equation!(acc, storage, model, eq, eq_s, force_as_stracer, time)
        tmp = sum(acc, dims = 1)
        active_entities = findall(i -> length(ST.deriv(tmp[i]).nzind) > 0, eachindex(tmp))

        # Determine what entities in the equation this force actually uses.
        setup_sparsity_struct_forces!(sparsity, extra_sparsity, nentity, neqs, npartials, active_entities, eqname, offset)
        offset += neqs*nentity
    end
    return sparsity
end

function setup_sparsity_struct_forces!(sparsity, extra_sparsity, nentity, neqs, npartials, active_entities, eqname, offset)
    S = Vector{Int}()
    # I = similar(S)
    rows = Vector{Vector{Int}}()
    cols = Vector{Vector{Int}}()
    if haskey(extra_sparsity, eqname)
        for c in extra_sparsity[eqname]
            push!(active_entities, c)
        end
    end
    for i in active_entities
        push!(S, i)
        I = Int[]
        # TODO: I / rows is not really needed here?
        for e in 1:neqs
            e_ix = Jutul.alignment_linear_index(i, e, nentity, neqs, EquationMajorLayout())
            push!(I, e_ix)
        end
        push!(rows, I)

        J = Int[]
        for p in 1:npartials
            p_ix = Jutul.alignment_linear_index(i, p, nentity, npartials, EquationMajorLayout())
            push!(J, p_ix)
        end
        push!(cols, J)
    end
    sparsity[eqname] = (entity = S, rows = rows, cols = cols, dims = (neqs, nentity))
end

function determine_cross_term_sparsity_forces(model, subforces, extra_sparsity, offset = 0)
    out = OrderedDict{Symbol, Any}()
    if !isnothing(subforces)
        sparsity = OrderedDict{Symbol, Any}()
        for (eqname, eq) in model.equations
            entity = Jutul.associated_entity(eq)
            nentity = number_of_entities(model, eq)
            neqs = number_of_equations_per_entity(model, eq)
            npartials = Jutul.number_of_partials_per_entity(model, entity)

            setup_sparsity_struct_forces!(sparsity, extra_sparsity, nentity, neqs, npartials, Int[], eqname, offset)
            offset += neqs*nentity
        end
        for (fname, force) in pairs(subforces)
            # Should all be the same
            out[fname] = sparsity
        end
    end
    return out
end

function evaluate_force_gradient(X, model::SimulationModel, storage, parameters, forces, config, forceno, time, dt)
    mname = :Model
    mmodel = MultiModel((Model = model,))
    return evaluate_force_gradient(
        X, mmodel, storage,
        Dict(mname => parameters),
        Dict(mname => forces),
        Dict(mname => config),
        forceno,
        time,
        dt
    )
end

function unique_forces_and_mapping(allforces, timesteps; eachstep = false)
    function force_steps(forces, dt)
        return ([forces], [1:length(dt)], 1)
    end

    function force_steps(forces::Vector, dt)
        unique_forces = unique(forces)
        num_unique_forces = length(unique_forces)
        force_map = Vector{Vector{Int}}(undef, num_unique_forces)
        for i in 1:num_unique_forces
            force_map[i] = Int[]
        end
        for (i, uf) in enumerate(unique_forces)
            for (j, f) in enumerate(forces)
                if f == uf
                    push!(force_map[i], j)
                end
            end
        end
        @assert sum(length, force_map) == length(forces) "Expected all forces to be accounted for ($(sum(length, force_map)) ≠ $(length(forces)))."
        return (unique_forces, force_map, num_unique_forces)
    end
    if eachstep
        if !(allforces isa Vector)
            allforces = [copy(allforces) for _ in timesteps]
        end
        unique_forces = deepcopy(allforces)
        forces_to_timestep = collect(eachindex(timesteps))
        timesteps_to_forces = copy(forces_to_timestep)
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

function setup_adjoint_forces_storage(model, allforces, timesteps;
        n_objective = nothing,
        use_sparsity = true,
        eachstep = false,
        targets = force_targets(model),
        forces_map = missing,
        state0 = setup_state(model),
        parameters = setup_parameters(model)
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
    storage[:unique_forces] = unique_forces
    storage[:forces_to_timestep] = forces_to_timestep
    storage[:timestep_to_forces] = timesteps_to_forces
    storage[:forces_map] = forces_map
    storage[:forces_gradient] = []
    storage[:forces_vector] = []
    storage[:forces_config] = []
    storage[:forces_sparsity] = []
    storage[:forces_jac] = []
    storage[:targets] = targets

    nvar = storage.n_forward
    for (i, force) in enumerate(unique_forces)
        X, config = vectorize_forces(force, model, targets)
        push!(storage[:forces_gradient], zeros(length(X)))
        push!(storage[:forces_vector], X)
        push!(storage[:forces_config], config)
        S_force = determine_sparsity_forces(model, force, X, config, parameters = parameters)
        push!(storage[:forces_sparsity], S_force)
        push!(storage[:forces_jac], sparse(Int[], Int[], Float64[], nvar, length(X)))
    end
    return storage
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
    storage = setup_adjoint_forces_storage(model, allforces, timesteps; state0 = state0, parameters = parameters, kwarg...)
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
        kwarg...
    )
    states, timesteps, step_ix = expand_to_ministeps(states, reports)
    unique_forces, forces_to_timestep, timesteps_to_forces, = storage[:forces_map]
    if allforces isa Vector
        allforces = allforces[step_ix]
    end

    fg = storage[:forces_gradient]
    fv = storage[:forces_vector]
    fc = storage[:forces_config]
    if init
        t = storage[:targets]
        for (forceno, force) in enumerate(unique_forces)
            fv[forceno], fc[forceno] = vectorize_forces(force, model, t)
        end
    end
    for g in fg
        @. g = 0.0
    end

    N = length(timesteps)
    @assert N == length(states)
    # Do sparsity detection if not already done.
    update_objective_sparsity!(storage, G, states, timesteps, allforces, :forward)
    for i in N:-1:1
        forceno = timesteps_to_forces[step_ix[i]]
        # Unpack stuff for this force in particular
        forces = unique_forces[forceno]
        config = fc[forceno]
        out = fg[forceno]
        X = fv[forceno]

        s, s0, s_next = Jutul.state_pair_adjoint_solve(state0, states, i, N)
        λ, t, dt, forces = Jutul.next_lagrange_multiplier!(storage, i, G, s, s0, s_next, timesteps, forces)
        J = evaluate_force_gradient(X, model, storage, parameters, forces, config, forceno, t, timesteps[i])
        mul!(out, J', λ, 1.0, 1.0)
    end

    return solve_adjoint_forces_retval(storage, model)
end

function solve_adjoint_forces_retval(storage, model::SimulationModel)
    dX = storage[:forces_gradient]
    dforces = map(
        (forces, out, config) -> devectorize_forces(forces, model, out, config),
        storage[:unique_forces], dX, storage[:forces_config]
    )
    return (dforces, storage[:timestep_to_forces], dX)
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
                    push!(row_labels, "$fname.$subname")
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
    force_adj_storage = setup_adjoint_forces_storage(
        model,
        forces,
        dt,
        state0 = state0,
        parameters = parameters
    )

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

