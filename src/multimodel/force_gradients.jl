function vectorize_forces(forces, model::MultiModel, variant = :all; T = Float64)
    offset = 0
    config = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        submodel = model[k]
        if variant isa AbstractDict
            subvariant = variant[k]
        else
            subvariant = variant
        end
        sublengths = vectorization_lengths(forces[k], submodel, subvariant)
        config[k] = (
            lengths = sublengths,
            offsets = [offset+1],
            meta = Dict{Symbol, Any}(),
            variant = variant
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
    for k in submodels_symbols(model)
        submodel = model[k]
        subforces = forces[k]
        subconfig = config[k]

        extra = Dict()
        for ct in evaluate_force_gradient_get_crossterms(model, k)
            is_self = ct.target == k
            if is_self
                ekey = ct.target_equation
                eq = submodel.equations[ekey]
                ct_cells = Jutul.cross_term_entities(ct.cross_term, eq, submodel)
            else
                @assert ct.source == k
                ekey = ct.source_equation
                eq = submodel.equations[ekey]
                ct_cells = Jutul.cross_term_entities_source(ct.cross_term, eq, submodel)
            end
            if !haskey(extra, ekey)
                extra[ekey] = Int[]
            end
            for c in ct_cells
                push!(extra[ekey], c)
            end
        end
        subX = X[subconfig.offsets[1]:subconfig.offsets[1]+sum(subconfig.lengths)-1]
        subparameters = parameters[k]
        sparsity[k] = determine_sparsity_forces(submodel, subforces, subX, subconfig; parameters = subparameters, extra_sparsity = extra)
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

function evaluate_force_gradient(X, model::MultiModel, storage, parameters, forces, config, forceno, time; row_offset = 0, col_offset = 0)
    offset_var = 0
    offset_x = 0
    for (k, m) in pairs(model.models)
        ndof = number_of_degrees_of_freedom(m)
        nl = sum(config[k].lengths)
        X_k = view(X, (offset_x+1):(offset_x+nl))
        evaluate_force_gradient_inner(X_k, model, k, storage, parameters[k], forces[k], config[k], forceno, time, row_offset)
        offset_var += ndof
        offset_x += nl
    end
    return storage[:forces_jac][forceno]
end

function evaluate_force_gradient_inner(X, multi_model::MultiModel, model_key::Symbol, storage, parameters, forces, config, forceno, time, row_offset::Int)
    dt = NaN # TODO: Fix.

    getstate(k) = as_value(storage.forward.storage.state[k])
    getstate0(k) = as_value(storage.forward.storage.state0[k])

    function add_in_cross_term!(acc, state_t, state0_t, model_t, ct_pair, eq, dt)
        ct = ct_pair.cross_term
        if ct_pair.target == model_key
            impact = cross_term_entities(ct, eq, model_t)
            sgn = 1.0
            other = ct_pair.source
        else
            impact = cross_term_entities_source(ct, eq, model_t)
            if symmetry(ctp.cross_term) == CTSkewSymmetry()
                sgn = -1.0
            else
                sgn = 1.0
            end
            other = ct_pair.target
        end
        model_s = multi_model[other]
        state_s = getstate(other)
        state0_s = getstate0(other)
        N = length(impact)
        # TODO: apply_force_to_cross_term!
        v = zeros(eltype(acc), size(acc, 1), N)
        for i in 1:N
            prepare_cross_term_in_entity!(i, state_t, state0_t, state_s, state0_s, model_t, model_s, ct, eq, dt)
            ldisc = local_discretization(ct, i)
            v_i = view(v, :, i)
            update_cross_term_in_entity!(v_i, i, state_t, state0_t, state_s, state0_s, model_t, model_s, ct, eq, dt, ldisc)
        end
        increment_equation_entries!(acc, model, v, impact, N, sgn)
        return acc
    end

    model = multi_model[model_key]
    J = storage[:forces_jac][forceno]
    # Find maximum width
    offsets = config.offsets
    if sum(config.lengths) == 0
        return J
    end
    state = getstate(model_key)
    state0 = getstate0(model_key)

    nvar = storage.n_forward
    sparsity = storage[:forces_sparsity][forceno]

    npartials = maximum(diff(offsets))
    sample = Jutul.get_ad_entity_scalar(1.0, npartials, 1)
    # Initialize acc + forces with that size ForwardDiff.Dual
    T = typeof(sample)
    X_ad = Vector{T}(undef, length(X))
    for fno in 1:(length(offsets)-1)
        local_index = 1
        for j in offsets[fno]:(offsets[fno+1]-1)
            X_ad[j] = Jutul.get_ad_entity_scalar(X[j], npartials, local_index)
            local_index += 1
        end
    end
    mstorage = JutulStorage()
    mstorage[:state] = state
    mstorage[:state0] = state0
    setup_storage_model(mstorage, model)
    forces_ad = devectorize_forces(forces, model, X_ad, config)
    update_before_step!(mstorage, model, dt, forces, time = time)
    offsets = config.offsets
    fno = 1
    for (fname, force) in pairs(forces_ad)
        offset = offsets[fno] - 1
        np = offsets[fno+1] - offsets[fno] # check off by one
        if haskey(sparsity, model_key)
            S = sparsity[model_key][fname]
        else
            S = sparsity[fname]
        end
        for (eqname, S) in pairs(S)
            eq = model.equations[eqname]
            acc = zeros(T, S.dims)
            eq_s = missing
            Jutul.apply_forces_to_equation!(acc, mstorage, model, eq, eq_s, force, time)
            cts = evaluate_force_gradient_get_crossterms(multi_model, model_key, eqname)
            for ct_pair in cts
                add_in_cross_term!(acc, state, state0, model, ct_pair, eq, dt)
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
    return (dforces, dX)
end

function devectorize_forces(forces, model::MultiModel, X, config; offset = 0)
    new_forces = Dict{Symbol, Any}()
    for k in submodels_symbols(model)
        submodel = model[k]
        subforces = forces[k]
        subconfig = config[k]
        n = sum(subconfig.lengths)
        subX = X[(offset+1):(offset+n)]
        nf = devectorize_forces(subforces, submodel, subX, subconfig)
        new_forces[k] = setup_forces(submodel; nf...)
        offset += n
    end
    return new_forces
end