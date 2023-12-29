struct MRSTWrapMesh <: FiniteVolumeMesh
    data
    N::Matrix
    nc::Int
    nf::Int
    tags::MeshEntityTags{Int}
end

function MRSTWrapMesh(G, N = nothing)
    @assert haskey(G, "cells")
    @assert haskey(G, "faces")
    if !haskey(G, "nodes")
        @warn "Grid is missing nodes. Coarse grid? Plotting will not work."
    end
    G = convert_to_immutable_storage(deepcopy(G))
    if isnothing(N)
        N = Int64.(G.faces.neighbors)
        internal_faces = (N[:, 2] .> 0) .& (N[:, 1] .> 0)
        N = collect(N[internal_faces, :]')
    end
    nf = size(N, 2)
    nc = G.cells.num
    g = MRSTWrapMesh(G, N, nc, nf, tags)
    initialize_entity_tags!(g)
    return g
end

function grid_dims_ijk(g::MRSTWrapMesh)
    return tuple(Int.(g.data.cartDims)...)
end

function cell_dims(g::MRSTWrapMesh, pos::Integer)
    fp = g.data.cells.facePos
    faces = Int.(g.data.cells.faces[Int(fp[pos]):Int(fp[pos+1]-1)])
    fc = g.data.faces.centroids[faces, :]
    dim = size(fc, 2)
    tmp = zeros(dim)
    for i in 1:dim
        x = fc[:, i]
        tmp[i] = maximum(x) - minimum(x)
    end
    return tuple(tmp...)
end

function cell_ijk(g::MRSTWrapMesh, base_index::Integer)
    if haskey(g.data.cells, :indexMap)
        imap =  g.data.cells.indexMap
        t = Int(imap[base_index])
    else
        t = base_index
    end
    nx, ny, nz = grid_dims_ijk(g)
    # (z-1)*nx*ny + (y-1)*nx + x
    x = mod(t - 1, nx) + 1
    y = mod((t - x) ÷ nx, ny) + 1
    leftover = (t - x - (y-1)*nx)
    z = (leftover ÷ (nx*ny)) + 1
    return (x, y, z)
end


function declare_entities(g::MRSTWrapMesh)
    return [
        (entity = Cells(), count = number_of_cells(g)),
        (entity = Faces(), count = number_of_faces(g)),
    ]
end

function Base.show(io::IO, m::MRSTWrapMesh)
    nc = number_of_cells(m)
    nf = number_of_faces(m)
    print(io, "MRSTWrapMesh with $nc cells and $nf faces.")
end

function tpfv_geometry(g::MRSTWrapMesh)
    exported = g.data
    faces = exported.faces
    cells = exported.cells
    cell_centroids = copy((cells.centroids)')

    N = g.N
    N_raw = Int64.(faces.neighbors)
    internal_faces = (vec(N_raw[:, 2]) .> 0) .& (vec(N_raw[:, 1]) .> 0)
    # Neighborship might mismatch the actual grid neighborships if the MRST grid
    # comes from any number of routines that try to create both a geometrically
    # valid grid and the TPFA neighborship defined somewhere else. We do some
    # checks on that plus the geometry fields, otherwise we insert NaN and
    # assume that the result is still somewhat useful.
    nf = size(N, 2)
    nf_all = size(N_raw, 1)
    check_face(k) = haskey(faces, k) && size(faces[k], 1) == nf_all
    N_ok = all(N_raw[internal_faces, :]' == g.N)
    self_consistent = N_ok && check_face(:areas) && check_face(:normals) && check_face(:centroids)
    if self_consistent
        face_centroids = copy((faces.centroids[internal_faces, :])')
        face_areas = vec(faces.areas[internal_faces])
        face_normals = faces.normals[internal_faces, :]./face_areas
        face_normals = copy(face_normals')
    else
        @assert eltype(N)<:Integer
        @assert size(N, 1) == 2
        dim = size(cell_centroids, 1)
        fake_vec() = repeat([NaN], dim, nf)
        fake_scalar() = repeat([NaN], nf)
        face_centroids = fake_vec()
        face_normals = fake_vec()
        face_areas = fake_scalar()
    end
    V = cells.volumes

    return TwoPointFiniteVolumeGeometry(N, face_areas, V, face_normals, cell_centroids, face_centroids)
end

dim(t::MRSTWrapMesh) = Int64(t.data.griddim)
number_of_cells(t::MRSTWrapMesh) = t.nc
number_of_faces(t::MRSTWrapMesh) = t.nf
neighbor(t::MRSTWrapMesh, f, i) = Int64(t.data.faces.neighbors[f, i])


function plot_primitives(mesh::MRSTWrapMesh, plot_type; kwarg...)
    # By default, no plotting is supported
    if plot_type == :mesh
        out = triangulate_mesh(mesh; kwarg...)
    elseif plot_type == :meshscatter
        out = meshscatter_primitives(mesh; kwarg...)
    else
        out = nothing
    end
    return out
end

function triangulate_mesh(m::Dict)
    mm = MRSTWrapMesh(m)
    return triangulate_mesh(mm)
end

function triangulate_mesh(m::MRSTWrapMesh; is_depth = true, outer = false)
    G = m.data
    d = dim(m)

    # Indirection map cells -> faces
    fpos = Int64.(G.cells.facePos)
    faces = Int64.(G.cells.faces)
    # Indirection map faces -> nodes
    nodePos = Int64.(G.faces.nodePos)
    gnodes = Int64.(G.faces.nodes)
    npts = G.nodes.coords

    function face_nodes(f)
        return Int64.(gnodes[nodePos[f]:nodePos[f+1]-1])
    end
    function cell_faces(c)
        return Int64.(faces[fpos[c]:fpos[c+1]-1])
    end
    function cyclical_tesselation(n)
        c = ones(Int64, n)
        # Create a triangulation of face, assuming convexity
        # Each tri is two successive points on boundary connected to centroid
        start = 2
        stop = n+1
        l = start:stop
        r = [stop, (start:stop-1)...]

        return hcat(l, c, r)
    end
    pts = Vector{Matrix{Float64}}()
    tri = Vector{Matrix{Int64}}()
    cell_index = Vector{Int64}()
    face_index = similar(cell_index)
    offset = 0
    nc = Int64(G.cells.num)
    if d == 1
        nodes = G.nodes.coords
        nn = size(nodes, 1)
        @assert nn == nc + 1
        offset = 0
        for cell = 1:nc
            x0 = nodes[cell]
            x1 = nodes[cell+1]

            push!(tri, [1 2 4; 1 4 3] .+ offset)
            push!(pts, [x0 0.0; x1 0.0; x0 1.0; x1 1.0])
            N = 4
            for i in 1:N
                push!(cell_index, cell) 
            end
            offset += N
        end
    elseif d == 2
        # For each cell, rotate around and add all nodes and triangles that include the center cell
        ccent = G.cells.centroids
        for cell = 1:nc
            center = ccent[cell, :]
            local_pts = [center']
            local_tri = []
            cf = cell_faces(cell)
            for (i, f) in enumerate(cf)
                local_nodes = face_nodes(f)
                @assert length(local_nodes) == 2
                if G.faces.neighbors[f, 1] == cell
                    node_ix = 1
                else
                    node_ix = 2
                end
                push!(local_pts, npts[local_nodes[node_ix], :]')
            end
            local_pts = vcat(local_pts...)
            local_tri = vcat(local_tri...)
            n = size(local_pts, 1) - 1
            local_tri = cyclical_tesselation(n)

            # Out
            push!(pts, local_pts)
            push!(tri, local_tri .+ offset)

            new_vert_count = n + 1
            for _ in 1:new_vert_count
                push!(cell_index, cell)
            end
            offset = offset + new_vert_count
        end
    else
        @assert d == 3
        # Find boundary faces
        N = Int64.(G.faces.neighbors)
        if outer
            exterior = findall(vec(any(N .== 0, dims = 2)))
            cells = sum(N[exterior, :], dims = 2)
            active = exterior
        else
            cells = 1:nc
            active = 1:nc
        end
        fcent = G.faces.centroids
        for i in active
            cell = cells[i]
            for f in cell_faces(cell)        
                center = fcent[f, :]
                # Grab local nodes
                local_nodes = face_nodes(f)
                edge_pts = npts[local_nodes, :]
                # Face centroid first, then nodes
                local_pts = [center'; edge_pts]
                if is_depth
                    # TODO: Reverse zaxis for Makie instead of doing it here.
                    local_pts[:, 3] *= -1
                end
                n = length(local_nodes)
                local_tri = cyclical_tesselation(n)
                # Out
                push!(pts, local_pts)
                push!(tri, local_tri .+ offset)

                new_vert_count = n + 1
                for _ in 1:new_vert_count
                    push!(cell_index, cell)
                    push!(face_index, f)
                end
                offset = offset + new_vert_count
            end
        end
    end
    pts = plot_flatten_helper(pts)
    tri = plot_flatten_helper(tri)

    mapper = (
                Cells = (cell_data) -> cell_data[cell_index],
                Faces = (face_data) -> face_data[face_index],
                indices = (Cells = cell_index, Faces = face_index)
              )
    return (points = pts, triangulation = tri, mapper = mapper)
end
