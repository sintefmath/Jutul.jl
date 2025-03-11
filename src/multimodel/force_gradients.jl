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
        for ct in model.cross_terms
            is_self = ct.target == k
            is_self_symm = ct.source == k && Jutul.has_symmetry(ct.cross_term)
            if is_self || is_self_symm
                if is_self
                    ekey = ct.target_equation
                    eq = submodel.equations[ekey]
                    ct_cells = Jutul.cross_term_entities(ct.cross_term, eq, submodel)
                end
                if is_self_symm
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
        end
        subX = X[subconfig.offsets[1]:subconfig.offsets[1]+sum(subconfig.lengths)-1]
        subparameters = parameters[k]
        sparsity[k] = determine_sparsity_forces(submodel, subforces, subX, subconfig; parameters = subparameters, extra_sparsity = extra)
    end
    return sparsity
end

function evaluate_force_gradient(X, model::MultiModel, storage, parameters, forces, config, forceno, time; row_offset = 0, col_offset = 0)
    offset_var = 0
    offset_x = 0
    J = storage[:forces_jac][forceno]
    for (k, m) in pairs(model.models)
        ndof = number_of_degrees_of_freedom(m)
        nl = sum(config[k].lengths)
        X_k = view(X, (offset_x+1):(offset_x+nl))
        evaluate_force_gradient(X_k, m, storage, parameters[k], forces[k], config[k], forceno, time,
            row_offset = offset_var,
            col_offset = offset_var,
            model_key = k
        )
        offset_var += ndof
        offset_x += nl
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

    dX = storage[:forces_vector]
    dforces = map(
        inner_retval,
        storage[:unique_forces], storage[:forces_config], dX
    )
    return (dforces, dX)
end
