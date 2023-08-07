export subdomain, submap_cells, subforces, subforce, global_map, coarse_neighborhood
export SimplePartition, SimpleMultiModelPartition, number_of_subdomains, entity_subset

abstract type AbstractDomainPartition end

struct SimplePartition{E, P, S} <: AbstractDomainPartition
    partition::P
    subsets::S
    entity::E
end

function SimplePartition(p; entity = Cells(), subsets = missing)
    np = maximum(p)
    for i in 1:np
        @assert any(x -> x == i, p)
    end
    @assert minimum(p) == 1
    if ismissing(subsets)
        subsets = map(
            index -> findall(isequal(index), p),
            1:np
            )
    else
        @assert length(subsets) == np
        for (i, subset) in enumerate(subsets)
            for c in subset
                # TODO: Check validity here.
            end
        end
    end
    return SimplePartition(p, subsets, entity)
end

main_partition(sp::SimplePartition) = sp
main_partition_label(sp::SimplePartition) = nothing

number_of_subdomains(sp::SimplePartition) = maximum(sp.partition)
entity_subset(sp, index) = entity_subset(sp, index, Cells())
entity_subset(sp::SimplePartition, index, e::Cells) = sp.subsets[index]

struct SimpleMultiModelPartition <: AbstractDomainPartition
    partition::Dict{Symbol, Any}
    main_symbol::Symbol
    function SimpleMultiModelPartition(p, m)
        new(p, m)
    end
end
main_partition(mp::SimpleMultiModelPartition) = mp.partition[mp.main_symbol]
number_of_subdomains(mp::SimpleMultiModelPartition) = number_of_subdomains(main_partition(mp))
main_partition_label(mp::SimpleMultiModelPartition) = mp.main_symbol

subdiscretization(disc, ::TrivialGlobalMap) = disc

function subgrid

end

function subdomain(d::DiscretizedDomain, indices; entity = Cells(), variables_always_active = false, kwarg...)
    grid = physical_representation(d)
    disc = d.discretizations

    N = get_neighborship(grid)
    t = @elapsed begin
        cells, faces, is_boundary = submap_cells(global_map(d), N, indices; kwarg...)
        mapper = FiniteVolumeGlobalMap(cells, faces, is_boundary, variables_always_active = variables_always_active)
        sg = subgrid(grid, cells = cells, faces = faces)
        d = Dict()
        for k in keys(disc)
            d[k] = subdiscretization(disc[k], sg, mapper)
        end
        subdisc = convert_to_immutable_storage(d)
    end
    @debug "Created domain with $(length(indices)) cells in $t seconds."
    return DiscretizedDomain(sg, subdisc, global_map = mapper)
end

function submap_cells(gmap, N, indices; nc = maximum(N), buffer = 0, excluded = [])
    @assert buffer == 0 || buffer == 1
    has_buffer_zone = buffer > 0

    facelist, facepos = get_facepos(N)
    nf = size(N, 2)
    cell_active = BitArray(false for x = 1:nc)
    cell_is_bnd = BitArray(false for x = 1:nc)
    face_active = BitArray(false for x = 1:nf)

    interior_cells = BitArray(false for x = 1:nc)
    interior_cells[indices] .= true

    insert_face!(f) = face_active[f] = true
    function insert_cell!(gc, is_bnd)
        cell_active[gc] = true
        cell_is_bnd[gc] = is_bnd
    end
    # Loop over all cells and add them to the global list
    for gc in indices
        # Include global cell
        partition_boundary = false
        for fi in facepos[gc]:facepos[gc+1]-1
            face = facelist[fi]
            l, r = N[1, face], N[2, face]
            # If both cells are in the interior, we add the face to the list
            if interior_cells[l] && interior_cells[r]
                insert_face!(face)
            else
                # Cell is on the boundary of the partition since it has an exterior face.
                # This might be the global boundary of the subdomain, if buffer == 0
                partition_boundary = true
            end
        end
        is_bnd = partition_boundary && buffer == 0
        insert_cell!(gc, is_bnd)
    end
    # If we have a buffer, we also need to go over and add the buffer cells
    if has_buffer_zone
        for gc in indices
            # Also add the neighbors, if not already present
            for fi in facepos[gc]:facepos[gc+1]-1
                face = facelist[fi]
                l, r = N[1, face], N[2, face]
                if l == gc
                    other = r
                else
                    other = l
                end
                if in(other, excluded)
                    continue
                end
                # If a bounded cell is not in the global list of interior cells
                # and not in the current set of processed cells, we add it and
                # flag as a global boundary
                insert_face!(face)
                # if !in(other, cells) && !in(other, indices)
                if !cell_active[other] && !interior_cells[other]
                    # This cell is now guaranteed to be on the boundary.
                    insert_cell!(other, true)
                end
            end
        end
    end
    if has_buffer_zone
        # TODO: This is not order preserving.
        cells = findall(cell_active)
        is_boundary = cell_is_bnd[cells]
        for i in 1:length(is_boundary)
            inside = in(cells[i], indices)
            if is_boundary[i] && buffer == 1
                @assert !inside
            else
                @assert inside
            end
        end
    else
        cells = copy(indices)
        is_boundary = BitVector([false for i in 1:length(cells)])
        if gmap isa FiniteVolumeGlobalMap
            for (i, c) in enumerate(cells)
                if gmap.cell_is_boundary[c]
                    is_boundary[i] = true
                end
            end
        end
    end
    faces = findall(face_active)
    return (cells = cells, faces = faces, is_boundary = is_boundary)
end

function subforces(forces, submodel)
    D = Dict{Symbol, Any}()
    if isnothing(forces)
        return nothing
    else
        for k in keys(forces)
            D[k] = subforce(forces[k], submodel)
        end
        return NamedTuple(pairs(D))
    end
end

function subforces(forces, submodel::MultiModel)
    D = Dict{Symbol, Any}()
    for k in keys(forces)
        if haskey(submodel.models, k)
            D[k] = subforces(forces[k], submodel.models[k])
        end
    end
    return D
end

subforce(::Nothing, model) = nothing
subforce(t, model) = copy(t) # A bit dangerous.

# export subparameters
# subparameters(model, param) = deepcopy(param)

# function subparameters(model::MultiModel, param)
#     D = typeof(param)()
#     for (k, v) in param
#         if haskey(model.models, k)
#             D[k] = deepcopy(v)
#         end
#     end
#     return D
# end

function coarse_neighborhood(p, submodel)
    M = global_map(submodel.domain)
    cells = M.cells
    return unique(p.partition[cells])
end

function coarse_neighborhood(p::SimpleMultiModelPartition, submodel::MultiModel)
    s = p.main_symbol
    return coarse_neighborhood(main_partition(p), submodel.models[s])
end