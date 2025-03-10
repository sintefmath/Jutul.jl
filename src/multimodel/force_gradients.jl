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
        subX = X[subconfig.offsets[1]:subconfig.offsets[1]+sum(subconfig.lengths)-1]
        subparameters = parameters[k]
        sparsity[k] = determine_sparsity_forces(submodel, subforces, subX, subconfig; parameters = subparameters)
    end
    return sparsity
end

function devectorize_forces(forces, model::MultiModel, X, config)
    new_forces = OrderedDict{Symbol, Any}()
    lengths = config.lengths
    offset = 0
    ix = 1
    error()
    for (k, v) in pairs(forces)
        if isnothing(v)
            continue
        end
        n_i = lengths[ix]
        X_i = view(X, (offset+1):(offset+n_i))
        new_forces[k] = devectorize_force(v, model, X_i, config.meta[k], k, config.variant)
        offset += n_i
        ix += 1
    end
    return new_forces
end

