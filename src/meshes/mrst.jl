struct MRSTWrapMesh <: AbstractJutulMesh
    data
    function MRSTWrapMesh(G)
        @assert haskey(G, "cells")
        @assert haskey(G, "faces")
        if !haskey(G, "nodes")
            @warn "Grid is missing nodes. Coarse grid? Plotting will not work."
        end
        G = convert_to_immutable_storage(deepcopy(G))
        return new(G)
    end
end

function tpfv_geometry(g::MRSTWrapMesh)
    exported = g.data
    faces = exported.faces
    cells = exported.cells

    N = Int64.(faces.neighbors)
    internal_faces = (N[:, 2] .> 0) .& (N[:, 1] .> 0)
    N = copy(N[internal_faces, :]')

    face_centroids = copy((faces.centroids[internal_faces, :])')
    face_areas = vec(faces.areas[internal_faces])
    face_normals = faces.normals[internal_faces, :]./face_areas
    face_normals = copy(face_normals')
    cell_centroids = copy((cells.centroids)')
    V = cells.volumes

    return TwoPointFiniteVolumeGeometry(N, face_areas, V, face_normals, cell_centroids, face_centroids)
end

dim(t::MRSTWrapMesh) = Int64(t.data.griddim)
number_of_cells(t::MRSTWrapMesh) = Int64(t.data.cells.num)
number_of_faces(t::MRSTWrapMesh) = Int64(t.data.faces.num)
neighbor(t::MRSTWrapMesh, f, i) = Int64(t.data.faces.neighbors[f, i])

function triangulate_outer_surface(m::Dict)
    mm = MRSTWrapMesh(m)
    return triangulate_outer_surface(mm)
end

function triangulate_outer_surface(m::MRSTWrapMesh, is_depth = true)
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
    pts = []
    tri = []
    cell_index = []
    face_index = []
    offset = 0
    if d == 2
        # For each cell, rotate around and add all nodes and triangles that include the center cell
        # error("Not implemented.")
        nc = Int64(G.cells.num)
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
            push!(cell_index, repeat([cell], new_vert_count))
            offset = offset + new_vert_count
        end

    else
        @assert d == 3
        # Find boundary faces
        N = Int64.(G.faces.neighbors)
        exterior = findall(vec(any(N .== 0, dims = 2)))
        cells = sum(N[exterior, :], dims = 2)
        fcent = G.faces.centroids

        for i in 1:length(exterior)
            cell = cells[i]
            for f in cell_faces(cell)
                local_tri = []
                local_pts = []
        
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
                push!(cell_index, repeat([cell], new_vert_count))
                push!(face_index, repeat([f], new_vert_count))

                offset = offset + new_vert_count
            end
        end
    end
    pts = vcat(pts...)
    tri = vcat(tri...)

    cell_index = vcat(cell_index...)
    face_index = vcat(face_index...)

    mapper = (
                Cells = (cell_data) -> cell_data[cell_index],
                Faces = (face_data) -> face_data[face_index],
                indices = (Cells = cell_index, Faces = face_index)
              )
    return (pts, tri, mapper)
end
