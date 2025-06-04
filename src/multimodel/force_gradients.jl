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
    vectorize_forces!(v, model, config, forces, update_config = true)
    return (v, config)
end

function vectorize_forces!(v, model::MultiModel, config, forces; kwarg...)
    for k in submodels_symbols(model)
        vectorize_forces!(v, model[k], config[k], forces[k]; kwarg...)
    end
    return v
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