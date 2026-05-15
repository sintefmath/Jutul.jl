export UnstructuredMesh
include("types.jl")
include("utils.jl")
include("geometry.jl")
include("plotting.jl")
include("refinement.jl")

dim(t::UnstructuredMesh{D}) where D = D::Int

function get_neighborship(G::UnstructuredMesh; internal = true)
    if internal
        nf = number_of_faces(G)
        N = zeros(Int, 2, nf)
        for (i, lr) in enumerate(G.faces.neighbors)
            N[1, i] = lr[1]
            N[2, i] = lr[2]
        end
    else
        N = G.boundary_faces.neighbors
    end
    return N
end

function grid_dims_ijk(g::UnstructuredMesh)
    # For unstructured mesh, we just return number of cells in x, and 1 in y and z
    ncells = number_of_cells(g)
    return (ncells, 1, 1)
end

function grid_dims_ijk(g::UnstructuredMesh{D, CartesianIndex{D}}) where D
    dims = Tuple(g.structure)
    if D == 1
        nx, = dims
        ny = nz = 1
    elseif D == 2
        nx, ny = dims
        nz = 1
    else
        @assert D == 3
        nx, ny, nz = dims
    end
    return (nx, ny, nz)
end

function cell_ijk(g)
    return i -> cell_ijk(g, i)
end

function cell_ijk(g::UnstructuredMesh{D, CartesianIndex{D}}, index::Integer) where D
    nx, ny, nz = grid_dims_ijk(g)
    if isnothing(g.cell_map)
        @assert number_of_cells(g) == nx*ny*nz
        t = index
    else
        t = g.cell_map[index]
    end
    # (z-1)*nx*ny + (y-1)*nx + x
    x = mod(t - 1, nx) + 1
    y = mod((t - x) ÷ nx, ny) + 1
    leftover = (t - x - (y-1)*nx)
    z = (leftover ÷ (nx*ny)) + 1
    return (x, y, z)
end

function cell_index(g::UnstructuredMesh, pos::Tuple; throw = true)
    nx, ny, nz = grid_dims_ijk(g)
    x, y, z = cell_ijk(g, pos)
    @assert x > 0 && x <= nx
    @assert y > 0 && y <= ny
    @assert z > 0 && z <= nz
    index = (z-1)*nx*ny + (y-1)*nx + x
    if isnothing(g.cell_map)
        @assert number_of_cells(g) == nx*ny*nz
        t = index
    else
        t = findfirst(isequal(index), g.cell_map)
        if isnothing(t)
            if throw
                error("Cell $pos not found in active set")
            else
                return nothing
            end
        end
    end
    return t
end

function cell_dims(g::UnstructuredMesh, pos)
    index = cell_index(g, pos)
    # Pick the nodes
    T = eltype(g.node_points)
    minv = Inf .+ zero(T)
    maxv = -Inf .+ zero(T)
    for face_set in [g.faces, g.boundary_faces]
        for face in face_set.cells_to_faces[index]
            for node in face_set.faces_to_nodes[face]
                pt = g.node_points[node]
                minv = min.(pt, minv)
                maxv = max.(pt, maxv)
            end
        end
    end
    Δ = maxv - minv
    @assert all(x -> x > 0, Δ) "Cell dimensions were zero? Computed $Δ for cell $index."
    return Tuple(Δ)
end

function plot_primitives(mesh::UnstructuredMesh, plot_type; kwarg...)
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
