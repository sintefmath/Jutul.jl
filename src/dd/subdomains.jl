export subdomain, submap_cells, subforces, subforce, global_map
export SimplePartition, SimpleMultiModelPartition, number_of_subdomains, entity_subset

abstract type AbstractDomainPartition end

struct SimplePartition <: AbstractDomainPartition
    partition
    entity
    function SimplePartition(p; entity = Cells())
        for i in 1:maximum(p)
            @assert any(x -> x == i, p)
        end
        @assert minimum(p) == 1
        new(p, entity)
    end
end
number_of_subdomains(sp::SimplePartition) = maximum(sp.partition)
entity_subset(sp, index, entity = Cells()) = entity_subset(sp, index, entity)
entity_subset(sp::SimplePartition, index, e::Cells) = findall(sp.partition .== index)

global_map(domain::DiscretizedDomain) = domain.global_map
global_map(domain) = TrivialGlobalMap()

struct SimpleMultiModelPartition <: AbstractDomainPartition
    partition
    main_symbol::Symbol
    function SimpleMultiModelPartition(p, m)
        new(p, m)
    end
end
main_partition(mp::SimpleMultiModelPartition) = mp.partition[mp.main_symbol]
number_of_subdomains(mp::SimpleMultiModelPartition) = number_of_subdomains(main_partition(mp))


"Local face -> global face (full set)"
global_face(f, ::TrivialGlobalMap) = f
"Local cell -> global cell (full set)"
global_cell(c, ::TrivialGlobalMap) = c

"Global cell -> local cell (full set)"
local_cell(c, ::TrivialGlobalMap) = c
cell_is_boundary(c, ::TrivialGlobalMap) = false

"Global face -> local face (full set)"
local_face(f, ::TrivialGlobalMap) = f
"Local cell in full set -> inner cell (or zero)"
interior_cell(c, ::TrivialGlobalMap) = c

"Inner cell to local cell (full set)"
full_cell(c, ::TrivialGlobalMap) = c
full_cell(c, m::FiniteVolumeGlobalMap) = m.inner_to_full_cells[c]

global_cell_inside_domain(c, m) = true
global_cell_inside_domain(c, m::FiniteVolumeGlobalMap) = any(isequal(c), m.cells)

interior_face(f, m) = f

map_to_active(V, domain::DiscretizedDomain, entity) = map_to_active(V, domain, domain.global_map, entity)
map_to_active(V, domain, entity) = V
map_to_active(V, domain, m, entity) = V

function map_to_active(V, domain, m::FiniteVolumeGlobalMap, ::Cells)
    W = similar(V, 0)
    for i in V
        ix = interior_cell(i, m)
        if !isnothing(ix)
            push!(W, ix)
        end
    end
    return W
    # return filter(i -> m.cell_is_boundary[i], V)
end

map_ij_to_active(I, J, domain, entity) = map_ij_to_active(I, J, domain, domain.global_map, entity)

function map_ij_to_active(I, J, domain, m::FiniteVolumeGlobalMap, ::Cells)
    n = length(I)
    @assert n == length(J)
    In = copy(I)
    Jn = copy(J)
    active = Vector{Bool}(undef, n)
    for k in 1:n
        i_new = interior_cell(I[k], m)
        j_new = interior_cell(J[k], m)
        keep = !isnothing(i_new) && !isnothing(j_new)
        if keep
            In[k] = i_new
            Jn[k] = j_new
        end
        active[k] = keep
    end
    return (In[active], Jn[active])
end

map_ij_to_active(I, J, domain, m::TrivialGlobalMap, entity) = (I, J)

global_face(f, m::FiniteVolumeGlobalMap) = m.faces[f]
global_cell(c, m::FiniteVolumeGlobalMap) = m.cells[c]

# local_cell(c_global, m::FiniteVolumeGlobalMap{R}) where R = only(indexin(c_global, m.cells))::R
local_cell(c_global, m::FiniteVolumeGlobalMap{R}) where R = only(findfirst(isequal(c_global), m.cells))::R

local_face(f_global, m::FiniteVolumeGlobalMap) = only(indexin(f_global, m.faces))

cell_is_boundary(c, m::FiniteVolumeGlobalMap) = m.cell_is_boundary[c]


function interior_cell(c, m::FiniteVolumeGlobalMap)
    c_i = m.full_to_inner_cells[c]
    return c_i == 0 ? nothing : c_i
end
# interior_cell(c, m::FiniteVolumeGlobalMap) = only(indexin(c, m.inner_to_full_cells))

#global_map

# Specialize cells, leave faces be (probably already filtered)
active_cells(model) = active_entities(model.domain, Cells())

active_entities(d, m::FiniteVolumeGlobalMap, ::Cells) = m.inner_to_full_cells
active_entities(d, m::FiniteVolumeGlobalMap, f::Faces) = 1:count_entities(d, f)

active_entities(d::DiscretizedDomain, entity) = active_entities(d, d.global_map, entity)
active_entities(d::DiscretizedDomain, ::TrivialGlobalMap, entity) = 1:count_entities(d, entity)


# TODO: Probably a bit inefficient
count_active_entities(d, m, e) = length(active_entities(d, m, e))

subdiscretization(disc, ::TrivialGlobalMap) = disc

function subdomain(d::DiscretizedDomain, indices; entity = Cells(), buffer = 0)
    grid = d.grid
    disc = d.discretizations

    N = grid.neighborship
    t = @elapsed begin
        cells, faces, is_boundary = submap_cells(N, indices, buffer = buffer, use_fast = true)
        mapper = FiniteVolumeGlobalMap(cells, faces, is_boundary)
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

function submap_cells(N, indices; nc = maximum(N), buffer = 0, use_fast = false)
    @assert buffer == 0 || buffer == 1
    if use_fast
        out = submap_cells_fast(N, indices, nc, buffer)
    else
        out = submap_cells_slow(N, indices, nc, buffer)
    end
    return out
end

function submap_cells_fast(N, indices, nc = maximum(N), buffer = 0)

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
    if buffer == 1
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
    cells = findall(cell_active)
    is_boundary = cell_is_bnd[cells]
    for i in 1:length(is_boundary)
        inside = in(cells[i], indices)
        if is_boundary[i]
            @assert !inside
        else
            @assert inside
        end
    end
    faces = findall(face_active)
    return (cells = cells, faces = faces, is_boundary = is_boundary)
end

function submap_cells_slow(N, indices, nc = maximum(N), buffer = 0)
    cells = Vector{Integer}()
    faces = Vector{Integer}()
    is_boundary = Vector{Bool}()

    facelist, facepos = get_facepos(N)

    function insert_face!(f)
        if !in(f, faces)
            push!(faces, f)
        end
    end
    function insert_cell!(gc, is_bnd)
        push!(cells, gc)
        push!(is_boundary, is_bnd)
    end
    # Loop over all cells and add them to the global list
    for gc in indices
        # Include global cell
        partition_boundary = false
        for fi in facepos[gc]:facepos[gc+1]-1
            face = facelist[fi]
            l, r = N[1, face], N[2, face]
            # If both cells are in the interior, we add the face to the list
            if in(l, indices) && in(r, indices)
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
    if buffer == 1
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
                # If a bounded cell is not in the global list of interior cells and not in the current
                # set of processed cells, we add it and flag as a global boundary
                insert_face!(face)
                if !in(other, cells) && !in(other, indices)
                    # This cell is now guaranteed to be on the boundary.
                    insert_cell!(other, true)
                end
            end
        end
    end
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
subforce(t, model) = t # A bit dangerous.

export subparameters
subparameters(model, param) = deepcopy(param)

function subparameters(model::MultiModel, param)
    D = typeof(param)()
    for (k, v) in param
        if haskey(model.models, k)
            D[k] = deepcopy(v)
        end
    end
    return D
end