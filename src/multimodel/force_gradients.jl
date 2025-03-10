function vectorize_forces(forces, model::MultiModel, variant = :all; T = Float64, offset = 0)
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

function vectorization_lengths(forces, model::MultiModel, variant = :all)
    fvals = values(forces)
    lengths = zeros(Int, length(fvals))
    for (i, force) in enumerate(fvals)
        if !isnothing(force)
            lengths[i] = vectorization_length(force, variant)
        end
    end
    return lengths
end
