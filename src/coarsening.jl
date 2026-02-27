abstract type AbstractCoarseningFunction end

struct CoarsenByVolumeAverage <: AbstractCoarseningFunction end

function inner_apply_coarsening_function(finevals, fine_indices, op::CoarsenByVolumeAverage, coarse, fine, row, name, entity)
    subvols = fine[:volumes][fine_indices]
    return sum(finevals.*subvols)/sum(subvols)
end

struct CoarsenByHarmonicAverage <: AbstractCoarseningFunction end

function inner_apply_coarsening_function(finevals, fine_indices, op::CoarsenByHarmonicAverage, coarse, fine, row, name, entity)
    invvals = 1.0./finevals
    return length(invvals)/sum(invvals)
end

struct CoarsenByArithmeticAverage <: AbstractCoarseningFunction end

function inner_apply_coarsening_function(finevals, fine_indices, op::CoarsenByArithmeticAverage, coarse, fine, row, name, entity)
    return sum(finevals)/length(finevals)
end

struct CoarsenByFirstValue <: AbstractCoarseningFunction end

function inner_apply_coarsening_function(finevals, fine_indices, op::CoarsenByFirstValue, coarse, fine, row, name, entity)
    return finevals[1]
end

struct CoarsenByLargestCount <: AbstractCoarseningFunction end

function inner_apply_coarsening_function(finevals, fine_indices, op::CoarsenByLargestCount, coarse, fine, row, name, entity)
    uvals = unique(finevals)
    counts = Dict{eltype(finevals), Int}()
    for v in uvals
        counts[v] = 0
    end
    for v in finevals
        counts[v] += 1
    end
    return findmax(counts)[2]
end

struct CoarsenBySum <: AbstractCoarseningFunction end

function inner_apply_coarsening_function(finevals, fine_indices, op::CoarsenBySum, coarse, fine, row, name, entity)
    return sum(finevals)
end

struct CoarsenByMaximum <: AbstractCoarseningFunction end

function inner_apply_coarsening_function(finevals, fine_indices, op::CoarsenByMaximum, coarse, fine, row, name, entity)
    return maximum(finevals)
end

struct CoarsenByMinimum <: AbstractCoarseningFunction end

function inner_apply_coarsening_function(finevals, fine_indices, op::CoarsenByMinimum, coarse, fine, row, name, entity)
    return minimum(finevals)
end

function apply_coarsening_function!(coarsevals, finevals, op, coarse::DataDomain, fine::DataDomain, name, entity::Union{Cells, Faces}; coarse_to_cells = missing)
    CG = physical_representation(coarse)
    function block_indices(CG, block, ::Cells)
        if ismissing(coarse_to_cells)
            return findall(isequal(block), CG.partition)
        else
            return coarse_to_cells[block]
        end
    end
    function block_indices(CG, block, ::Faces)
        return CG.coarse_faces_to_fine[block]
    end
    function block_indices(CG, block, ::BoundaryFaces)
        return CG.coarse_boundary_to_fine[block]
    end
    ncoarse = count_entities(coarse, entity)
    if finevals isa AbstractVector
        for block in 1:ncoarse
            ix = block_indices(CG, block, entity)
            coarsevals[block] = inner_apply_coarsening_function(view(finevals, ix), ix, op, coarse, fine, 1, name, entity)
        end
    else
        for block in 1:ncoarse
            ix = block_indices(CG, block, entity)
            for j in axes(coarsevals, 1)
                coarsevals[j, block] = inner_apply_coarsening_function(view(finevals, j, ix), ix, op, coarse, fine, j, name, entity)
            end
        end
    end
    return coarsevals
end

function coarsen_data_domain(D::DataDomain, partition;
        functions = Dict(),
        default = CoarsenByArithmeticAverage(),
        default_other = CoarsenByLargestCount(),
        kwarg...
    )
    for (k, v) in pairs(kwarg)
        functions[k] = v
    end
    g = physical_representation(D)
    cg = CoarseMesh(g, partition)
    cD = DataDomain(cg)
    coarse_to_cells = map(i -> findall(isequal(i), cg.partition), 1:maximum(cg.partition))
    for name in keys(D)
        if !haskey(cD, name)
            val = D[name]
            e = Jutul.associated_entity(D, name)
            Te = eltype(val)
            if !(e in (Cells(), Faces(), BoundaryFaces(), NoEntity(), nothing))
                # Other entities are not supported yet.
                continue
            end
            if isnothing(e) || e isa NoEntity
                # No idea about coarse dims, just copy
                coarseval = deepcopy(val)
            elseif val isa AbstractVecOrMat
                ne = count_entities(cg, e)
                if val isa AbstractVector
                    coarseval = zeros(Te, ne)
                else
                    coarseval = zeros(Te, size(val, 1), ne)
                end
                if eltype(Te)<:AbstractFloat
                    f = get(functions, name, default)
                else
                    f = get(functions, name, default_other)
                end
                coarseval = apply_coarsening_function!(coarseval, val, f, cD, D, name, e, coarse_to_cells = coarse_to_cells)
            else
                # Don't know what's going on
                coarseval = deepcopy(val)
            end
            cD[name, e] = coarseval
        end
    end
    return cD
end
