export MinimalTPFAGrid
export get_cell_faces, get_facepos, get_cell_neighbors
export number_of_cells, number_of_faces, number_of_half_faces

export transfer, get_1d_reservoir

import Base.eltype

abstract type PorousMediumGrid <: TervGrid end
abstract type ReservoirGrid <: PorousMediumGrid end
# TPFA grid
"Minimal struct for TPFA-like grid. Just connection data and pore-volumes"
struct MinimalTPFAGrid{R<:AbstractFloat, I<:Integer} <: ReservoirGrid
    pore_volumes::AbstractVector{R}
    neighborship::AbstractArray{I}
    function MinimalTPFAGrid(pv, N)
        nc = length(pv)
        pv::AbstractVector
        @assert size(N, 1) == 2  "Two neighbors per face"
        if length(N) > 0
            @assert minimum(N) > 0   "Neighborship entries must be positive."
            @assert maximum(N) <= nc "Neighborship must be limited to number of cells."
        end
        @assert all(pv .> 0)     "Pore volumes must be positive"
        new{eltype(pv), eltype(N)}(pv, N)
    end
end


function number_of_cells(G::ReservoirGrid)
    return length(G.pore_volumes)
end

function number_of_faces(G)
    size(get_neighborship(G), 2)
end

function declare_entities(G::MinimalTPFAGrid)
    c = (entity = Cells(), count = number_of_cells(G)) # Cells equal to number of pore volumes
    f = (entity = Faces(), count = number_of_faces(G)) # Faces
    return [c, f]
end

function transfer(context::SingleCUDAContext, grid::MinimalTPFAGrid)
    pv = transfer(context, grid.pore_volumes)
    N = transfer(context, grid.neighborship)

    return MinimalTPFAGrid(pv, N)
end

function get_cell_faces(N, nc = nothing)
    # Create array of arrays where each entry contains the faces of that cell
    t = eltype(N)
    if length(N) == 0
        cell_faces = ones(t, 1)
    else
        if isnothing(nc)
            nc = maximum(N)
        end
        cell_faces = [Vector{t}() for i = 1:nc]
        for i in 1:size(N, 1)
            for j = 1:size(N, 2)
                push!(cell_faces[N[i, j]], j)
            end
        end
        # Sort each of them
        for i in cell_faces
            sort!(i)
        end
    end
    return cell_faces
end

function get_cell_neighbors(N, nc = maximum(N), includeSelf = true)
    # Find faces in each array
    t = typeof(N[1])
    cell_neigh = [Vector{t}() for i = 1:nc]
    for i in 1:size(N, 2)
        push!(cell_neigh[N[1, i]], N[2, i])
        push!(cell_neigh[N[2, i]], N[1, i])
    end
    # Sort each of them
    for i in eachindex(cell_neigh)
        loc = cell_neigh[i]
        if includeSelf
            push!(loc, i)
        end
        sort!(loc)
    end
    return cell_neigh
end

function get_facepos(N)
    if length(N) == 0
        t = eltype(N)
        faces = zeros(t, 0)
        facePos = ones(t, 2)
    else
        cell_faces = get_cell_faces(N)
        counts = [length(x) for x in cell_faces]
        facePos = cumsum([1; counts])
        faces = reduce(vcat, cell_faces)
    end
    return (faces, facePos)
end


function get_1d_reservoir(nc; L = 1, perm = 9.8692e-14, # 0.1 darcy
                         poro = 0.1, area = 1, fuse_flux = false,
                         z_max = nothing)
    @assert nc > 1 "Must have at least two cells."
    nf = nc-1
    N = vcat((1:nf)', (2:nc)')
    dx = L/nc
    cell_centroids = vcat((dx/2:dx:L-dx/2)', ones(2, nc))
    face_centroids = vcat((dx:dx:L-dx)', ones(2, nf))
    face_areas = ones(nf)
    face_normals = vcat(ones(1, nf), zeros(2, nf))

    function expand(x, nc)
        repeat([x], nc)
    end
    function expand(x::AbstractVector, nc)
        x
    end
    perm = expand(perm, nc)
    if isa(perm, AbstractVector)
        perm = copy(perm')
    end
    volumes = repeat([area.*dx], nc)

    pv = poro.*volumes
    nc = length(pv)

    @debug "Data unpack complete. Starting transmissibility calculations."
    # Deal with face data
    T_hf = compute_half_face_trans(cell_centroids, face_centroids, face_normals, face_areas, perm, N)
    T = compute_face_trans(T_hf, N)
    G = MinimalTPFAGrid(pv, N)
    if isnothing(z_max)
        z = nothing
        g = nothing
    else
        dz = z_max/nc
        z = (dz/2:dz:z_max-dz/2)'
        g = gravity_constant
    end

    if fuse_flux
        ft = DarcyMassMobilityFlowFused()
    else
        ft = DarcyMassMobilityFlow()
    end
    flow = TwoPointPotentialFlow(SPU(), TPFA(), ft, G, T, z, g)
    disc = (mass_flow = flow,)
    D = DiscretizedDomain(G, disc)
    return D
end
