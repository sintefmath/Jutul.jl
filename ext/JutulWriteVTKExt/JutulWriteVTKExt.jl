# NOTE: This extension was coded using AI agents (GitHub Copilot).
module JutulWriteVTKExt

using Jutul
using WriteVTK
using WriteVTK: vtk_grid, vtk_save, MeshCell, VTKPolyhedron, VTKCellTypes
using ProgressMeter

"""
    Jutul.export_mesh_vtu(mesh, filename; folder = ".", point_data = NamedTuple(), cell_data = NamedTuple())

Export a Jutul `UnstructuredMesh`/`JutulMesh`-compatible mesh to VTK unstructured
format using WriteVTK.jl.

`filename` should be given **without** an extension — the correct extension
(`.vtu` or `.pvd`) is inferred automatically from `cell_data`. Use `folder` to
control the output directory (defaults to the current directory).

The mesh is expected to expose the following fields (as in Jutul's
`UnstructuredMesh`):
- `node_points`
- `faces.cells_to_faces`, `faces.faces_to_nodes`, `faces.neighbors`
- `boundary_faces.cells_to_faces`, `boundary_faces.faces_to_nodes`

The `cell_data` argument can be:
- A `NamedTuple` or `Dict` mapping names to arrays (one value per cell, or a
  matrix with one column per cell), representing a single state.
- A reservoir state `Dict` with keys like `:Pressure`, `:Saturations`, etc.
  (as returned by a single simulation timestep).
- A `Vector` of such state dicts (as returned by `simulate_reservoir`),
  in which case a `.pvd` collection is written with one `.vtu` file per
  timestep.

Matrix fields (e.g. `Saturations` with shape `nphases × ncells`) are
automatically split into one scalar field per row, named `Saturations_1`,
`Saturations_2`, etc.

When the mesh has `z_is_depth = true` (z increases downward, as is common in
reservoir models), the z coordinates are negated on export so that the mesh
appears correctly oriented in ParaView (z-up convention). Set `flip_z = false`
to export raw coordinates instead.

Returns the list of written file paths from `vtk_save`.
"""
function Jutul.export_mesh_vtu(mesh, filename::AbstractString; folder = ".", flip_z = true, point_data = NamedTuple(), cell_data = NamedTuple())
    do_flip = flip_z && Jutul.mesh_z_is_depth(mesh)
    points = _points_matrix(mesh, do_flip)
    cells = _vtk_cells(mesh)
    nc = length(mesh.faces.cells_to_faces)

    base = joinpath(folder, _strip_extension(basename(filename)))
    mkpath(folder)

    # Determine if cell_data is a vector of states (one dict per timestep)
    if cell_data isa AbstractVector && length(cell_data) > 0 && first(cell_data) isa AbstractDict
        # Vector of states: write a PVD collection.
        pvd = WriteVTK.paraview_collection(base)
        @showprogress desc="Exporting states to VTK..." for (step, state) in enumerate(cell_data)
            step_filename = "$(base)_step$(lpad(step, 4, '0'))"
            vtk = _build_vtu(points, cells, nc, step_filename, point_data, state)
            pvd[Float64(step)] = vtk
        end
        return WriteVTK.vtk_save(pvd)
    else
        return vtk_save(_build_vtu(points, cells, nc, base, point_data, cell_data))
    end
end

function _strip_extension(filename)
    base, ext = splitext(filename)
    return isempty(ext) ? filename : base
end

function _build_vtu(points, cells, nc, filename, point_data, cell_data)
    vtk = vtk_grid(filename, points, cells)
    for (name, values) in pairs(point_data)
        vtk[string(name), WriteVTK.VTKPointData()] = values
    end
    _add_cell_data!(vtk, nc, cell_data)
    return vtk
end

function _add_cell_data!(vtk, nc, cell_data)
    for (name, values) in pairs(cell_data)
        _add_cell_field!(vtk, string(name), values, nc)
    end
end

function _add_cell_field!(vtk, name, values, nc)
    if values isa AbstractMatrix && eltype(values) <: Number && size(values, 2) == nc
        # Split each row into a separate scalar field to avoid VTK interpreting
        # rows as X/Y/Z vector components (which requires exactly 3 rows).
        nrows = size(values, 1)
        if nrows == 1
            vtk[name, WriteVTK.VTKCellData()] = Float64.(vec(values))
        else
            for row in 1:nrows
                vtk["$(name)_$(row)", WriteVTK.VTKCellData()] = Float64.(values[row, :])
            end
        end
    elseif values isa AbstractVector && eltype(values) <: Number && length(values) == nc
        vtk[name, WriteVTK.VTKCellData()] = Float64.(values)
    end
    # Skip fields whose element type is not a Number (e.g. BlackOilX) or whose
    # size doesn't match the cell count (e.g. face arrays, scalars).
end

function _points_matrix(mesh, flip_z = false)
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
    if flip_z && dim == 3
        @. pts[3, :] = -pts[3, :]
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
