# using Meshes, MeshViz

export MRSTWrapMesh, triangulate_outer_surface# , Polyogonal2DMesh

struct BasicFiniteVolumeGeometry
    areas
    volumes
    normals
    face_centers
    cell_centers
end

abstract type AbstractTervMesh end

dim(t::AbstractTervMesh) = 2

# struct Polyogonal2DMesh <: AbstractTervMesh
#     simple_mesh
# end

struct MRSTWrapMesh <: AbstractTervMesh
    data
    # neighbors
    # simple_mesh
    function MRSTWrapMesh(G)
        @assert haskey(G, "cells")
        @assert haskey(G, "faces")
        @assert haskey(G, "nodes")
        G = convert_to_immutable_storage(G)
        return new(G)

        # coord = G["nodes"]["coords"]
        # n = size(coord, 1)
        # d = Int64(G["griddim"])
        # # Convert coordinate array
        # if d == 3
        #     P = Meshes.Point3
        # else
        #     P = Meshes.Point2
        # end
        # pts = Vector{P}(undef, n)
        # for i = 1:n
        #     pts[i] = P(coord[i, :]...)
        # end
        # # Indirection map cells -> faces
        # facePos = Int64.(G["cells"]["facePos"])
        # faces = Int64.(G["cells"]["faces"])
        # # Indirection map faces -> nodes
        # nodePos = Int64.(G["faces"]["nodePos"])
        # gnodes = Int64.(G["faces"]["nodes"])
        # # Number of cells
        # nc = Int64(G["cells"]["num"])
        # nf = Int64(G["faces"]["num"])
        # conn = []
        # @info d
        # if d == 2
        #     for c = 1:nc
        #         tmp = []
        #         for (i, ix) in enumerate(facePos[c]:facePos[c+1]-1)
        #             f = faces[ix]
        #             npts = gnodes(nodePos[f]:nodePos[f+1]-1)
        #             if i == 1
        #                 push!(tmp, npts[1])
        #             end
        #             push!(tmp, npts[2])
        #         end
        #         push!(conn, Tuple(tmp))
        #     end
        # else
        #     for f = 1:nf
        #         local_nodes = gnodes[nodePos[f]:nodePos[f+1]-1]
        #         local_nodes = reverse(local_nodes)
        #         push!(conn, Tuple(local_nodes))
        #     end
        # end
        # display(conn)
        # C = connect.(conn)
        # m = SimpleMesh(pts, C)
        # new(G, nothing, m)
    end
end

dim(t::MRSTWrapMesh) = Int64(t.data.griddim)
number_of_cells(t::MRSTWrapMesh) = Int64(t.data.cells.num)
number_of_faces(t::MRSTWrapMesh) = Int64(t.data.faces.num)
neighbor(t::MRSTWrapMesh, f, i) = Int64(t.data.faces.neighbors[f, i])


cell_to_surface(m::AbstractTervMesh, celldata) = celldata

function cell_to_surface(m::MRSTWrapMesh, celldata)
    d = dim(m)
    if d == 2
        facedata = celldata
    else
        @assert d == 3
        nf = number_of_faces(m)
        facedata = zeros(nf)
        for i = 1:nf
            count = 0
            v = 0
            for j = 1:2
                cell = neighbor(m, i, j)
                if cell > 0
                    v += celldata[cell]
                    count += 1
                end
            end
            facedata[i] = v/count
        end
    end
    @info "Updated."
    return facedata
end


function triangulate_outer_surface(m::MRSTWrapMesh)
    G = m.data
    d = dim(m)
    if d == 2
        # For each cell, rotate around and add all nodes and triangles that include the center cell
        error("Not implemented.")
    else
        @assert d == 3
        # Find boundary faces
        N = Int64.(G.faces.neighbors)
        exterior = findall(vec(any(N .== 0, dims = 2)))
        cells = sum(N[exterior, :], dims = 2)
        ##
        # Indirection map cells -> faces
        fpos = Int64.(G.cells.facePos)
        faces = Int64.(G.cells.faces)
        # Indirection map faces -> nodes
        nodePos = Int64.(G.faces.nodePos)
        gnodes = Int64.(G.faces.nodes)
        npts = G.nodes.coords
        fcent = G.faces.centroids

        pts = []
        tri = []
        cell_index = []
        face_index = []
        offset = 0
        for i in 1:length(exterior)
            cell = cells[i]
            for fi in fpos[cell]:fpos[cell+1]-1
                local_tri = []
                local_pts = []
        
                f = faces[fi]
                center = fcent[f, :]
                # Grab local nodes
                local_nodes = Int64.(gnodes[nodePos[f]:nodePos[f+1]-1])
                edge_pts = npts[local_nodes, :]
                # Face centroid first, then nodes
                local_pts = [center'; edge_pts]
                n = length(local_nodes)
                c = ones(Int64, n)
                # Create a triangulation of face, assuming convexity
                # Each tri is two successive points on boundary connected to centroid
                start = 2
                stop = n+1
                l = start:stop
                r = [stop, (start:stop-1)...]

                local_tri = hcat(l, c, r)
                # Out
                push!(pts, local_pts)
                push!(tri, local_tri .+ offset)

                new_vert_count = n + 1
                push!(cell_index, repeat([cell], new_vert_count))
                push!(face_index, repeat([f], new_vert_count))

                offset = offset + new_vert_count
            end
        end
        pts = vcat(pts...)
        tri = vcat(tri...)

        cell_index = vcat(cell_index...)
        face_index = vcat(face_index...)
    end
    mapper = (
                Cells = (cell_data) -> cell_data[cell_index],
                Faces = (face_data) -> face_data[face_index],
                indices = (Cells = cell_index, Faces = face_index)
              )
    return (pts, tri, mapper)
end