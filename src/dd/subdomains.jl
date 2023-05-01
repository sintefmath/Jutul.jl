export subdomain, submap_cells, subforces, subforce, global_map, coarse_neighborhood
export SimplePartition, SimpleMultiModelPartition, number_of_subdomains, entity_subset

abstract type AbstractDomainPartition end

struct SimplePartition{E, P} <: AbstractDomainPartition
    partition::P
    entity::E
    function SimplePartition(p::T; entity = Cells()) where T
        for i in 1:maximum(p)
            @assert any(x -> x == i, p)
        end
        @assert minimum(p) == 1
        new{typeof(entity), T}(p, entity)
    end
end

number_of_subdomains(sp::SimplePartition) = maximum(sp.partition)
entity_subset(sp, index, entity = Cells()) = entity_subset(sp, index, entity)
entity_subset(sp::SimplePartition, index, e::Cells) = findall(sp.partition .== index)


struct SimpleMultiModelPartition <: AbstractDomainPartition
    partition
    main_symbol::Symbol
    function SimpleMultiModelPartition(p, m)
        new(p, m)
    end
end
main_partition(mp::SimpleMultiModelPartition) = mp.partition[mp.main_symbol]
number_of_subdomains(mp::SimpleMultiModelPartition) = number_of_subdomains(main_partition(mp))

subdiscretization(disc, ::TrivialGlobalMap) = disc

function subgrid

end

function subdomain(d::DiscretizedDomain, indices; entity = Cells(), variables_always_active = false, kwarg...)
    grid = physical_representation(d)
    disc = d.discretizations

    N = get_neighborship(grid)
    t = @elapsed begin
        cells, faces, is_boundary = submap_cells(N, indices; kwarg...)
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

function submap_cells(N, indices; nc = maximum(N), buffer = 0, excluded = [])
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
                # If a bounded cell is not in the global list of interior cells and not in the current
                # set of processed cells, we add it and flag as a global boundary
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
        is_boundary = cell_is_bnd[cells]
        for i in 1:length(is_boundary)
            inside = in(cells[i], indices)
            if is_boundary[i] && buffer == 1
                @assert !inside
            else
                @assert inside
            end
        end
        # TODO: This is not order preserving.
        cells = findall(cell_active)
    else
        cells = copy(indices)
        is_boundary = BitVector([false for i in 1:length(cells)])
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
        return convert_to_immutable_storage(D)
    end
end

function subforces(forces, submodel::MultiModel)
    D = Dict{Symbol, Any}()
    for k in keys(forces)
        if haskey(submodel.models, k)
            D[k] = subforces(forces[k], submodel.models[k])
        end
    end
    return convert_to_immutable_storage(D)
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