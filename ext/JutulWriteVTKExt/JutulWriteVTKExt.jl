# NOTE: This extension was coded using AI agents (GitHub Copilot).
module JutulWriteVTKExt

using Jutul
using WriteVTK
using WriteVTK: vtk_grid, vtk_save, MeshCell, VTKPolyhedron, VTKCellTypes

"""
    Jutul.export_mesh_vtu(mesh, filename; point_data = NamedTuple(), cell_data = NamedTuple())

Export a Jutul `UnstructuredMesh`/`JutulMesh`-compatible mesh to VTK unstructured
format (`.vtu`) using WriteVTK.jl.

The mesh is expected to expose the following fields (as in Jutul's
`UnstructuredMesh`):
- `node_points`
- `faces.cells_to_faces`, `faces.faces_to_nodes`, `faces.neighbors`
- `boundary_faces.cells_to_faces`, `boundary_faces.faces_to_nodes`

Returns the list of written file paths from `vtk_save`.
"""
function Jutul.export_mesh_vtu(mesh, filename::AbstractString; point_data = NamedTuple(), cell_data = NamedTuple())
    points = _points_matrix(mesh)
    cells = _vtk_cells(mesh)

    vtk = vtk_grid(filename, points, cells)
    for (name, values) in pairs(point_data)
        vtk[string(name), WriteVTK.VTKPointData()] = values
    end
    for (name, values) in pairs(cell_data)
        vtk[string(name), WriteVTK.VTKCellData()] = values
    end
    return vtk_save(vtk)
end

function _points_matrix(mesh)
    nnodes = length(mesh.node_points)
    nnodes == 0 && throw(ArgumentError("Mesh has no nodes"))

    dim = length(first(mesh.node_points))
    dim in (2, 3) || throw(ArgumentError("Only 2D and 3D meshes are supported, got dim = $dim"))

    pts = Matrix{Float64}(undef, dim, nnodes)
    for (i, p) in enumerate(mesh.node_points)
        for d in 1:dim
            pts[d, i] = Float64(p[d])
        end
    end
    return pts
end

function _vtk_cells(mesh)
    ncells = length(mesh.faces.cells_to_faces)
    ncells > 0 || throw(ArgumentError("Mesh has no cells"))

    dim = length(first(mesh.node_points))
    if dim == 2
        return [_vtk_cell_2d(mesh, c) for c in 1:ncells]
    else
        return [_vtk_cell_3d(mesh, c) for c in 1:ncells]
    end
end

function _vtk_cell_2d(mesh, cell::Int)
    edges = Tuple{Int, Int}[]

    for face in mesh.faces.cells_to_faces[cell]
        n1, n2 = _face_nodes_pair(mesh.faces.faces_to_nodes[face])
        push!(edges, (n1, n2))
    end
    for face in mesh.boundary_faces.cells_to_faces[cell]
        n1, n2 = _face_nodes_pair(mesh.boundary_faces.faces_to_nodes[face])
        push!(edges, (n1, n2))
    end

    conn = _polygon_connectivity_from_edges(edges)
    n = length(conn)

    if n == 3
        ctype = VTKCellTypes.VTK_TRIANGLE
    elseif n == 4
        ctype = VTKCellTypes.VTK_QUAD
    else
        ctype = VTKCellTypes.VTK_POLYGON
    end
    return MeshCell(ctype, conn)
end

function _vtk_cell_3d(mesh, cell::Int)
    faces = Vector{Vector{Int}}()

    for face in mesh.faces.cells_to_faces[cell]
        nodes = collect(Int, mesh.faces.faces_to_nodes[face])
        l, r = mesh.faces.neighbors[face]
        if r == cell
            reverse!(nodes)
        end
        push!(faces, nodes)
    end

    for face in mesh.boundary_faces.cells_to_faces[cell]
        push!(faces, collect(Int, mesh.boundary_faces.faces_to_nodes[face]))
    end

    conn = unique!(vcat(faces...))
    return VTKPolyhedron(conn, Tuple.(faces)...)
end

function _face_nodes_pair(face_nodes)
    length(face_nodes) == 2 || throw(ArgumentError("2D face must have exactly 2 nodes, got $(length(face_nodes))"))
    return (Int(face_nodes[1]), Int(face_nodes[2]))
end

function _polygon_connectivity_from_edges(edges::Vector{Tuple{Int, Int}})
    isempty(edges) && throw(ArgumentError("Cell had no edges"))

    adj = Dict{Int, Vector{Int}}()
    for (a, b) in edges
        push!(get!(adj, a, Int[]), b)
        push!(get!(adj, b, Int[]), a)
    end

    for (node, nbrs) in adj
        length(nbrs) == 2 || throw(ArgumentError("Expected manifold 2D cell boundary at node $node, got degree $(length(nbrs))"))
    end

    start = first(keys(adj))
    path = Int[start]

    prev = nothing
    current = start
    while true
        nbrs = adj[current]
        next = isnothing(prev) || nbrs[1] != prev ? nbrs[1] : nbrs[2]
        if next == start
            break
        end
        push!(path, next)
        prev, current = current, next

        if length(path) > length(edges) + 1
            throw(ArgumentError("Failed to construct polygon connectivity from edges"))
        end
    end

    return path
end

end # module JutulWriteVTKExt
