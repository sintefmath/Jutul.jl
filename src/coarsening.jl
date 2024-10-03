abstract type AbstractCoarseningFunction end

struct CoarsenByVolumeAverage <: AbstractCoarseningFunction end

function inner_apply_coarsening_function!(finevals, fine_indices, op::CoarsenByVolumeAverage, coarse, fine, name, entity)
    subvols = fine[:volumes][fine_indices]
    return sum(finevals.*subvols)/sum(subvols)
end

struct CoarsenByHarmonicAverage <: AbstractCoarseningFunction end

function inner_apply_coarsening_function!(finevals, fine_indices, op::CoarsenByHarmonicAverage, coarse, fine, name, entity)
    invvals = 1.0./finevals
    return length(invvals)/sum(invvals)
end

struct CoarsenByArithemticAverage <: AbstractCoarseningFunction end

function inner_apply_coarsening_function!(finevals, fine_indices, op::CoarsenByArithemticAverage, coarse, fine, name, entity)
    return sum(finevals)/length(finevals)
end

function apply_coarsening_function!(coarsevals, finevals, op, coarse::DataDomain, fine::DataDomain, name, entity::JutulEntity)
    CG = physical_representation(coarse)
    function block_indices(CG, block)
        findall(isequal(block), CG.partition)
    end
    ncoarse = count_entities(coarse, entity)
    if finevals isa AbstractVector
        for block in 1:ncoarse
            ix = block_indices(CG, block)
            coarsevals[block] = inner_apply_coarsening_function!(view(finevals, ix), ix, op, coarse, fine, name, entity)
        end
    else
        for block in 1:ncoarse
            ix = block_indices(CG, block)
            for j in axes(coarsevals, 1)
                coarsevals[j, block] = inner_apply_coarsening_function!(view(finevals, j, ix), ix, op, coarse, fine, name, entity)
            end
        end
    end
    return coarsevals
end

function coarsen_data_domain(D::DataDomain, partition;
        functions = Dict(),
        default = CoarsenByArithemticAverage(),
        kwarg...
    )
    for (k, v) in pairs(kwarg)
        functions[k] = v
    end
    g = physical_representation(D)
    cg = CoarseMesh(g, partition)
    cD = DataDomain(cg)
    for name in keys(D)
        if !haskey(cD, name)
            val = D[name]
            e = Jutul.associated_entity(D, name)
            if isnothing(e)
                # No idea about coarse dims
                coarseval = deepcopy(val)
            elseif val isa AbstractVecOrMat
                ne = count_entities(cg, e)
                Te = eltype(val)
                if val isa AbstractVector
                    coarseval = zeros(Te, ne)
                else
                    coarseval = zeros(Te, size(val, 1), ne)
                end
            else
                # Don't know what's going on
                coarseval = deepcopy(val)
            end
            # Need to coarsen.
            f = get(functions, name, default)
            cD[name] = apply_coarsening_function!(coarseval, val, f, cD, D, name, e)
        end
    end
    return cD
end
