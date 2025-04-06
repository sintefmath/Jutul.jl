function force_targets(model::MultiModel, arg...)
    out = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        out[k] = force_targets(model[k], arg...)
    end
    return out
end

function vectorize_forces(forces, model::MultiModel, targets = force_targets(model); T = Float64)
    offset = 0
    config = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        submodel = model[k]
        target = targets[k]
        sublengths = vectorization_lengths(forces[k], submodel, target)
        config[k] = (
            lengths = sublengths,
            offsets = [offset+1],
            meta = Dict{Symbol, Any}(),
            targets = target
        )
        offset += sum(sublengths)
    end
    # Actually vectorize
    n = offset
    v = Vector{T}(undef, n)
    for k in submodels_symbols(model)
        vectorize_forces!(v, model[k], config[k], forces[k])
    end
    return (v, config)
end

function determine_sparsity_forces(model::MultiModel, forces, X, config; parameters = setup_parameters(model))
    sparsity = Dict{Symbol, Any}()
    offset = 0
    cross_term_sparsity = Dict{Symbol, Dict{Symbol, Any}}()
    for k in submodels_symbols(model)
        cross_term_sparsity_model = Dict{Symbol, Any}()
        cross_term_sparsity[k] = cross_term_sparsity_model
        submodel = model[k]
        # TODO: At the moment we don't distinguish between forces since
        # information can propgate from any force onto the cross terms.
        extra = Dict{Symbol, Any}()
        for ct in evaluate_force_gradient_get_crossterms(model, k)
            is_self = ct.target == k
            if is_self
                other = ct.source
                ekey = ct.target_equation
                eq = submodel.equations[ekey]
                ct_cells = Jutul.cross_term_entities(ct.cross_term, eq, submodel)
            else
                @assert ct.source == k
                other = ct.target
                ekey = ct.source_equation
                eq = submodel.equations[ekey]
                ct_cells = Jutul.cross_term_entities_source(ct.cross_term, eq, submodel)
            end
            if !haskey(cross_term_sparsity_model, other)
                cross_term_sparsity_model[other] = Dict{Symbol, Vector{Int}}()
            end
            extra = cross_term_sparsity_model[other]
            if !haskey(extra, ekey)
                extra[ekey] = Int[]
            end
            for c in ct_cells
                push!(extra[ekey], c)
            end
        end
    end

    for k in submodels_symbols(model)
        submodel = model[k]
        subforces = forces[k]
        subconfig = config[k]
        subX = X[subconfig.offsets[1]:subconfig.offsets[1]+sum(subconfig.lengths)-1]
        subparameters = parameters[k]
        extra = Dict{Symbol, Vector{Int}}()
        for (other_k, v) in cross_term_sparsity[k]
            for (k, entities) in v
                if !haskey(extra, k)
                    extra[k] = Int[]
                end
                for r in entities
                    push!(extra[k], r)
                end
            end
        end
        self = determine_sparsity_forces(submodel, subforces, subX, subconfig;
            parameters = subparameters,
            extra_sparsity = extra
        )
        # Get those cross terms that have source equal to k (where the forces
        # could potentially influence)
        other = Dict{Symbol, Any}()
        for target in submodels_symbols(model)
            if target != k
                other[target] = Dict{Symbol, Any}()
                for source in keys(cross_term_sparsity[target])
                    if source == k
                        cts = cross_term_sparsity[target][source]
                        other[target] = determine_cross_term_sparsity_forces(model[target], subforces, cts)
                    end
                end
            end
        end
        # other = cross_term_sparsity[k]
        sparsity[k] = (self = self, other = other)
    end
    return sparsity
end

function evaluate_force_gradient_get_crossterms(model, k, equation = missing)
    crossterms = CrossTermPair[]
    for ct in model.cross_terms
        is_self = ct.target == k
        is_self_symm = ct.source == k && Jutul.has_symmetry(ct.cross_term)
        if is_self || is_self_symm
            if ismissing(equation)
                push!(crossterms, ct)
            else
                if is_self && ct.target_equation == equation
                    push!(crossterms, ct)
                elseif is_self_symm && ct.source_equation == equation
                    push!(crossterms, ct)
                end
            end
        end
    end
    return crossterms
end

function evaluate_force_gradient!(dobj_dgrad, X, objective, model::MultiModel, storage, parameters, forces, config, forceno, time, step_no::Int, dt)
    forces = devectorize_forces(forces, model, X, config)
    offset_var = 0
    offset_x = 0
    is_fake_multimodel = haskey(model.models, :Model) && !haskey(storage.forward.storage.state, :Model)

    mstorage = JutulStorage()
    mstate = JutulStorage()
    mstate0 = JutulStorage()
    for k in submodels_symbols(model)
        s = JutulStorage()
        if is_fake_multimodel
            s[:state] = as_value(storage.forward.storage.state)
            s[:state0] = as_value(storage.forward.storage.state0)
        else
            s[:state] = as_value(storage.forward.storage.state[k])
            s[:state0] = as_value(storage.forward.storage.state0[k])
        end
        setup_storage_model(s, model.models[k])
        mstate[k] = s[:state]
        mstate0[k] = s[:state0]
        mstorage[k] = s
    end
    mstorage[:state] = mstate
    mstorage[:state0] = mstate0
    sparsity = storage[:forces_sparsity][forceno]
    J = storage[:forces_jac][forceno]
    offsets = Dict{Symbol, Int}()
    offset = 0
    for (k, m) in pairs(model.models)
        offsets[k] = offset
        offset += number_of_degrees_of_freedom(m)
    end

    nz = nonzeros(J)
    @. nz = 0.0
    update_before_step!(mstorage, model, dt, forces, time = time)
    for (k, m) in pairs(model.models)
        nl = sum(config[k].lengths)
        X_k = view(X, (offset_x+1):(offset_x+nl))
        evaluate_force_gradient!_inner!(J, dobj_dgrad, X_k, objective, model, k, storage, mstorage, parameters[k], forces, config[k], sparsity, time, step_no, dt, offsets)
        offset_x += nl
    end
    return J
end

function evaluate_force_gradient!_inner!(J, dobj_dgrad, X, objective, multi_model::MultiModel, model_key::Symbol, storage, model_storage, parameters, multimodel_forces, config, sparsity, time, step_no, dt, model_offsets::Dict{Symbol, Int})
    function add_in_cross_term!(acc, state_t, state0_t, model_t, target_key::Symbol, ct_pair, eq_label::Symbol, dt)
        ct = ct_pair.cross_term
        is_self = ct_pair.target == target_key
        eq = model_t.equations[eq_label]
        if is_self
            impact = cross_term_entities(ct, eq, model_t)
            sgn = 1.0
            other = ct_pair.source
            @assert ct_pair.target_equation == eq_label
        else
            impact = cross_term_entities_source(ct, eq, model_t)
            if symmetry(ct_pair.cross_term) == CTSkewSymmetry()
                sgn = -1.0
            else
                sgn = 1.0
            end
            other = ct_pair.target
            @assert ct_pair.source_equation == eq_label
        end

        model_s = multi_model[other]
        state_s = model_storage[other].state
        state0_s = model_storage[other].state0

        N = length(impact)
        # TODO: apply_force_to_cross_term!
        v = zeros(eltype(acc), size(acc, 1), N)
        if is_self
            s_arg = (state_t, state0_t, state_s, state0_s, model_t, model_s)
            eq_for_ct = model_t.equations[ct_pair.target_equation]
        else
            s_arg = (state_s, state0_s, state_t, state0_t, model_s, model_t)
            eq_for_ct = model_s.equations[ct_pair.source_equation]
        end
        for i in 1:N
            prepare_cross_term_in_entity!(i, s_arg..., ct, eq_for_ct, dt)
            ldisc = local_discretization(ct, i)
            v_i = view(v, :, i)
            update_cross_term_in_entity!(v_i, i, s_arg..., ct, eq_for_ct, dt, ldisc)
        end
        increment_equation_entries!(acc, model, v, impact, N, sgn)
        return acc
    end
    if haskey(sparsity, model_key)
        sparsity = sparsity[model_key]
    end

    model = multi_model[model_key]
    # Find maximum width
    if sum(config.lengths) == 0
        return J
    end
    state = model_storage[model_key].state
    state0 = model_storage[model_key].state0
    is_fake_multimodel = haskey(multi_model.models, :Model)

    nvar = storage.n_forward
    offsets = config.offsets
    npartials = maximum(diff(offsets))

    fno = 1
    subforces = multimodel_forces[model_key]
    row_offset = model_offsets[model_key]

    for fname in keys(subforces)
        all_forces = deepcopy(multimodel_forces)
        forces_ad = devectorize_forces(deepcopy(subforces), model, X, config, ad_key = fname)
        force_ad = forces_ad[fname]
        all_forces[model_key] = forces_ad
        update_before_step!(model_storage, multi_model, dt, all_forces, time = time)
        sample = Jutul.get_ad_entity_scalar(1.0, npartials, 1, tag = fname)
        T = typeof(sample)

        offset = offsets[fno] - 1
        np = offsets[fno+1] - offsets[fno]
        if haskey(sparsity, :self)
            self_sparsity = sparsity.self[fname]
        else
            self_sparsity = sparsity[fname]
        end
        for (eqname, S) in pairs(self_sparsity)
            eq = model.equations[eqname]
            acc = zeros(T, S.dims)
            eq_s = missing
            Jutul.apply_forces_to_equation!(acc, model_storage[model_key], model, eq, eq_s, force_ad, time)
            cts = evaluate_force_gradient_get_crossterms(multi_model, model_key, eqname)
            for ct_pair in cts
                add_in_cross_term!(acc, state, state0, model, model_key, ct_pair, eqname, dt)
            end
            # Loop over entities that this force impacts
            for (entity, rows) in zip(S.entity, S.rows)
                for (i, row) in enumerate(rows)
                    val = acc[i, entity]
                    for p in 1:np
                        ∂ = val.partials[p]
                        J[row + row_offset, offset + p] = ∂
                    end
                end
            end
        end
        if haskey(sparsity, :other)
            for (other_model_key, other_sparsity) in pairs(sparsity.other)
                row_offset_other = model_offsets[other_model_key]

                other_model = multi_model[other_model_key]
                other_state = model_storage[other_model_key].state
                other_state0 = model_storage[other_model_key].state0
                if !haskey(other_sparsity, fname)
                    continue
                end
                for (eqname, S) in pairs(other_sparsity[fname])
                    eq = other_model.equations[eqname]
                    entity = associated_entity(eq)
                    ne = count_entities(other_model.domain, entity)
                    nper_e = number_of_equations_per_entity(other_model, eq)
                    acc = zeros(T, (nper_e, ne))
                    eq_s = missing
                    # Jutul.apply_forces_to_equation!(acc, model_storage[model_key], model, eq, eq_s, force_ad, time)
                    cts = evaluate_force_gradient_get_crossterms(multi_model, other_model_key, eqname)
                    for ct_pair in cts
                        add_in_cross_term!(acc, other_state, other_state0, other_model, other_model_key, ct_pair, eqname, dt)
                    end
                    # Loop over entities that this force impacts
                    for (entity, rows) in zip(S.entity, S.rows)
                        for (i, row) in enumerate(rows)
                            val = acc[i, entity]
                            for p in 1:np
                                ∂ = val.partials[p]
                                J[row + row_offset_other, offset + p] = ∂
                            end
                        end
                    end
                end
            end
        end
        if is_fake_multimodel
            vstate = model_storage.state.Model
        else
            vstate = model_storage.state
        end
        partial_obj = objective(multi_model, as_value(vstate), dt, step_no, all_forces)
        if !(partial_obj isa Float64)
            for (derno, gderno) in enumerate(offsets[fno]:offsets[fno+1]-1)
                dobj_dgrad[gderno] += partial_obj.partials[derno]
            end
        end
        fno += 1
    end

    return J
end

function solve_adjoint_forces_retval(storage, model::MultiModel)
    function inner_retval(forces, config, dX_f)
        retval = Dict{Symbol, Any}()
        offset = 0
        for (k, m) in pairs(model.models)
            f = forces[k]
            c = config[k]
            n = sum(c.lengths)
            retval[k] = devectorize_forces(f, m, dX_f[offset+1:offset+n], c)
            offset += n
        end
        return retval
    end

    dX = storage[:forces_gradient]
    dforces = map(
        inner_retval,
        storage[:unique_forces], storage[:forces_config], dX
    )
    return (dforces, storage[:timestep_to_forces], dX)
end

function devectorize_forces(forces, model::MultiModel, X, config; offset = 0, model_key = nothing, ad_key = nothing)
    if isnothing(model_key)
        @assert isnothing(ad_key)
    end
    new_forces = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        submodel = model[k]
        subforces = forces[k]
        subconfig = config[k]
        n = sum(subconfig.lengths)
        subX = X[(offset+1):(offset+n)]
        if model_key == k
            ad_key_model = ad_key
        else
            ad_key_model = nothing
        end
        nf = devectorize_forces(subforces, submodel, subX, subconfig, ad_key = ad_key_model)
        new_forces[k] = setup_forces(submodel; nf...)
        offset += n
    end
    return new_forces
end