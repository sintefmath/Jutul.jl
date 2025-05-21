export cell_dims, grid_dims_ijk, cell_ijk, cell_index

"""
    CartesianMesh(dims, [Δ, [origin]])

Create a Cartesian mesh with dimensions specified by the `Tuple` `dims`.

# Arguments
- `dims::Tuple`: Number of grid cells in each direction. For example, `(nx, ny)` will give a 2D grids with `nx` cells in the x-direction.
- `Δ::Tuple=Tuple(ones(length(dims)))`: Equal length to `dims`. First option: A
`Tuple` of scalars where each entry is the length of each cell in that
direction. For example, specifying `(Δx, Δy) for a uniform grid with each grid
cell having area of `Δx*Δy`. Second option: A `Tuple` of vectors where each
entry contains the cell sizes in the direction.
- `origin=zeros(length(dims))`: The origin of the first corner in the grid.

# Examples
Generate a uniform 3D mesh that discretizes a domain of 2 by 3 by 5 units with 3 by 5 by 2 cells:
```julia-repl
julia> CartesianMesh((3, 5, 2), (2.0, 3.0, 5.0))
CartesianMesh (3D) with 3x5x2=30 cells
```

Generate a non-uniform 2D mesh:
```julia-repl
julia> CartesianMesh((2, 3), ([1.0, 2.0], [0.1, 3.0, 2.5]))
CartesianMesh (2D) with 2x3x1=6 cells
```
"""
struct CartesianMesh{D, Δ, O, T} <: FiniteVolumeMesh
    "Tuple of dimensions (nx, ny, [nz])"
    dims::D
    "Either a tuple of scalars (uniform grid) or a tuple of vectors (non-uniform grid)"
    deltas::Δ
    "Coordinate of lower left corner"
    origin::O
    "Tags on cells/faces/nodes"
    tags::MeshEntityTags{T}
    function CartesianMesh(dims::Tuple, deltas_or_size::Union{Nothing, Tuple} = nothing; origin = nothing)
        dim = length(dims)
        if isnothing(deltas_or_size)
            deltas_or_size = Tuple(ones(dim))
        end
        if isnothing(origin)
            origin = zeros(dim)
        else
            @assert length(origin) == dim
        end
        function generate_deltas(deltas_or_size)
            deltas = Vector(undef, dim)
            for (i, D) = enumerate(deltas_or_size)
                if isa(D, AbstractFloat)
                    # Deltas are actually size of domain in each direction
                    deltas[i] = D/dims[i]
                else
                    # Deltas are the actual cell widths
                    @assert length(D) == dims[i]
                    deltas[i] = D
                end
            end
            return Tuple(deltas)
        end
        @assert length(deltas_or_size) == dim
        deltas = generate_deltas(deltas_or_size)
        tags = MeshEntityTags()
        g = new{typeof(dims), typeof(deltas), typeof(origin), Int}(dims, deltas, origin, tags)
        initialize_entity_tags!(g)
        return g
    end
end
Base.show(io::IO, g::CartesianMesh) = print(io, "CartesianMesh ($(dim(g))D) with $(join(grid_dims_ijk(g), "x"))=$(number_of_cells(g)) cells")

function CartesianMesh(v::Int, d = 1.0; kwarg...)
    if d isa Float64
        d = (d, )
    end
    return CartesianMesh((v, ), d; kwarg...)
end

dim(t::CartesianMesh) = length(t.dims)
number_of_cells(t::CartesianMesh) = prod(t.dims)
function number_of_faces(t::CartesianMesh)
    nx, ny, nz = grid_dims_ijk(t)
    return (nx-1)*ny*nz + (ny-1)*nx*nz + (nz-1)*ny*nx
end

export number_of_boundary_faces
function number_of_boundary_faces(G::CartesianMesh)
    nx, ny, nz = grid_dims_ijk(G)
    D = dim(G)
    if D == 1
        nbnd = 2
    elseif D == 2
        nbnd = 2*(nx + ny)
    else
        @assert D == 3
        nbnd = 2*(nx*ny + ny*nz + nz*nx)
    end
    return nbnd
end

"""
Lower corner for one dimension, without any transforms applied
"""
coord_offset(pos, δ::AbstractFloat) = (pos-1)*δ
coord_offset(pos, δ::Union{AbstractVector, Tuple}) = sum(δ[1:(pos-1)], init = 0.0)

"""
    cell_index(g, pos)

Get linear (scalar) index of mesh cell from provided IJK tuple `pos`.
"""
function cell_index(g, pos::Tuple; throw = true)
    nx, ny, nz = grid_dims_ijk(g)
    x, y, z = cell_ijk(g, pos)
    return (z-1)*nx*ny + (y-1)*nx + x
end

function cell_index(g, pos::Integer; throw = true)
    return pos
end

function lower_corner_3d(g, index)
    pos = cell_ijk(g, index)
    Δ = g.deltas
    f = (i) -> coord_offset(pos[i], Δ[i])
    return Tuple(map(f, 1:3))
end

"""
    cell_dims(g, pos)::Tuple

Get physical cell dimensions of cell with index `pos` for grid `g`.
"""
function cell_dims(g, pos)
    x, y, z = cell_ijk(g, pos)
    Δ = g.deltas
    return (get_delta(Δ, x, 1), get_delta(Δ, y, 2), get_delta(Δ, z, 3))
end

function float_type(g::CartesianMesh)
    Δ = g.deltas
    return Base.promote_type(map(eltype, Δ)..., eltype(g.origin))
end

function tpfv_geometry(g::CartesianMesh)
    Δ = g.deltas
    d = dim(g)
    T = float_type(g)

    nx, ny, nz = grid_dims_ijk(g)

    # Cell data first - volumes and centroids
    nc = nx*ny*nz
    V = zeros(T, nc)
    cell_centroids = zeros(T, d, nc)
    for x in 1:nx
        for y in 1:ny
            for z = 1:nz
                pos = (x, y, z)
                c = cell_index(g, pos)
                cdim  = cell_dims(g, pos)
                V[c] = prod(cdim)

                for i in 1:d
                    cell_centroids[i, c] = coord_offset(pos[i], Δ[i]) + cdim[i]/2 + g.origin[i]
                end
            end
        end
    end

    # Then face data:
    nf = number_of_faces(g)
    N = Matrix{Int}(undef, 2, nf)
    face_areas = Vector{T}(undef, nf)
    face_centroids = zeros(T, d, nf)
    face_normals = zeros(T, d, nf)

    function add_face!(N, face_areas, face_normals, face_centroids, x, y, z, D, pos)
        t = (x, y, z)
        index = cell_index(g, t)
        N[1, pos] = index
        N[2, pos] = cell_index(g, (x + (D == 1), y + (D == 2), z + (D == 3)))
        Δ  = cell_dims(g, t)
        # Face area
        A = 1
        for i in setdiff(1:3, D)
            A *= Δ[i]
        end
        face_areas[pos] = A
        face_normals[D, pos] = 1.0

        face_centroids[:, pos] = cell_centroids[:, index]
        # Offset by the grid size
        face_centroids[D, pos] += Δ[D]/2.0
    end
    # Note: The following loops are arranged to reproduce the MRST ordering.
    pos = 1
    # Faces with X-normal > 0
    for z = 1:nz
        for y in 1:ny
            for x in 1:(nx-1)
                add_face!(N, face_areas, face_normals, face_centroids, x, y, z, 1, pos)
                pos += 1
            end
        end
    end
    # Faces with Y-normal > 0
    for y in 1:(ny-1)
        for z = 1:nz
            for x in 1:nx
                add_face!(N, face_areas, face_normals, face_centroids, x, y, z, 2, pos)
                pos += 1
            end
        end
    end
    # Faces with Z-normal > 0
    for z = 1:(nz-1)
        for y in 1:ny
            for x in 1:nx
                add_face!(N, face_areas, face_normals, face_centroids, x, y, z, 3, pos)
                pos += 1
            end
        end
    end


    nbnd = number_of_boundary_faces(g)
    # Then fix the boundary
    boundary_neighbors = Vector{Int}(undef, nbnd)
    boundary_areas = Vector{T}(undef, nbnd)
    boundary_normals = zeros(T, d, nbnd)
    boundary_centroids = zeros(T, d, nbnd)

    function add_boundary_face!(N, face_areas, face_normals, face_centroids, x, y, z, D, pos, is_start)
        t = (x, y, z)
        index = cell_index(g, t)
        N[pos] = index
        Δ  = cell_dims(g, t)
        # Face area
        A = 1
        for i in setdiff(1:3, D)
            A *= Δ[i]
        end
        face_areas[pos] = A
        if is_start
            sgn = -1.0
        else
            sgn = 1.0
        end
        face_normals[D, pos] = sgn
        face_centroids[:, pos] = cell_centroids[:, index]
        # Offset by the grid size
        face_centroids[D, pos] += sgn*Δ[D]/2.0
    end

    pos = 1
    # x varies, z, y fixed
    for y in 1:ny
        for z = 1:nz
            for (x, is_start) in [(1, true), (nx, false)]
                add_boundary_face!(boundary_neighbors, boundary_areas, boundary_normals, boundary_centroids, x, y, z, 1, pos, is_start)
                pos += 1
            end
        end
    end
    if d > 1
        # y varies, x, z fixed
        for x = 1:nx
            for z in 1:nz
                for (y, is_start) in [(1, true), (ny, false)]
                    add_boundary_face!(boundary_neighbors, boundary_areas, boundary_normals, boundary_centroids, x, y, z, 2, pos, is_start)
                    pos += 1
                end
            end
        end
        if d > 2
            # z varies, x, y fixed
            for x = 1:nx
                for y in 1:ny
                    for (z, is_start) in [(1, true), (nz, false)]
                        add_boundary_face!(boundary_neighbors, boundary_areas, boundary_normals, boundary_centroids, x, y, z, 3, pos, is_start)
                        pos += 1
                    end
                end
            end
        end
    end

    return TwoPointFiniteVolumeGeometry(
        N,
        face_areas,
        V,
        face_normals,
        cell_centroids,
        face_centroids;
        boundary_areas = boundary_areas,
        boundary_normals = boundary_normals,
        boundary_centroids = boundary_centroids,
        boundary_neighbors = boundary_neighbors
        )
end

function get_neighborship(g::CartesianMesh; internal = true)
    # Expensive but correct
    geo = tpfv_geometry(g)
    if internal
        N = geo.neighbors
    else
        N = geo.boundary_neighbors
    end
    return N
end

function grid_dims_ijk(g)
    d = length(g.dims)
    if d == 1
        nx = g.dims[1]
        ny = nz = 1
    elseif d == 2
        nx, ny = g.dims
        nz = 1
    else
        @assert d == 3
        nx, ny, nz = g.dims
    end
    return (nx, ny, nz)
end

function cell_ijk(g, t::Tuple)
    d = length(t)
    if d == 1
        nx = t[1]
        ny = nz = 1
    elseif d == 2
        nx, ny = t
        nz = 1
    else
        @assert d == 3
        nx, ny, nz = t
    end
    return (nx, ny, nz)
end

function cell_ijk(g, t::Integer)
    nx, ny, nz = grid_dims_ijk(g)
    # (z-1)*nx*ny + (y-1)*nx + x
    x = mod(t - 1, nx) + 1
    y = mod((t - x) ÷ nx, ny) + 1
    leftover = (t - x - (y-1)*nx)
    z = (leftover ÷ (nx*ny)) + 1
    return (x, y, z)
end


function get_delta(Δ, index, d)
    if length(Δ) >= d
        δ = Δ[d]
        if isa(δ, AbstractFloat)
            v = δ
        else
            v = δ[index]
        end
    else
        v = 1.0
    end
    return v
end

function plot_primitives(mesh::CartesianMesh, plot_type; kwarg...)
    if plot_type == :mesh
        out = triangulate_mesh(mesh; kwarg...)
    else
        out = nothing
    end
    return out
end

function triangulate_mesh(m::CartesianMesh; outer = false)
    pts = []
    tri = []
    cell_ix = []
    face_index = []
    offset = 0
    d = dim(m)
    # nc = number_of_cells(m)
    function append_face!(pts, tri, cell_ix, t, v, offset)
        local_pts, local_tri = v
        n = size(local_pts, 1)
        push!(pts, local_pts)
        push!(tri, local_tri .+ offset)
        push!(cell_ix, repeat([cell_index(m, t)], n))
        return n
    end
    if d == 2
        nx, ny, = m.dims
        Δ = m.deltas
        for x = 1:nx
            for y = 1:ny
                t = (x, y, 1)
                dx, dy,  = cell_dims(m, t)
                x0 = coord_offset(x, Δ[1])
                y0 = coord_offset(y, Δ[2])

                local_pts = [x0      y0;
                             x0 + dx y0;
                             x0 + dx y0 + dy;
                             x0      y0 + dy]
                local_tri = [1 2 3; 3 4 1]
                push!(pts, local_pts)
                push!(tri, local_tri .+ offset)
                push!(cell_ix, repeat([cell_index(m, t)], 4))
                offset += 4
            end
        end
    else
        @assert d == 3
        nx, ny, nz = m.dims
        function get_surface(t, planar_dim, is_end)
            dx, dy, dz = cell_dims(m, t)
            x0, y0, z0 = lower_corner_3d(m, t)

            if planar_dim == 1
                x = x0
                if is_end
                    x += dx
                end
                local_pts = [x y0      z0;
                             x y0      z0 + dz
                             x y0 + dy z0 + dz;
                             x y0 + dy z0]
            elseif planar_dim == 2
                y = y0
                if is_end
                    y += dy
                end
                local_pts = [x0      y z0;
                             x0 + dx y z0
                             x0 + dx y z0 + dz;
                             x0      y z0 + dz]
            else
                z = z0
                if is_end
                    z += dz
                end
                local_pts = [x0      y0      z;
                             x0 + dx y0      z
                             x0 + dx y0 + dy z;
                             x0      y0 + dy z]
            end
            local_tri = [1 2 3; 3 4 1]
            return (local_pts, local_tri)
        end
        if outer
            for y in 1:ny
                for x in 1:nx
                    for z in 1:nz
                        t = (x, y, z)
                        if x == 1 || x == nx || include_all
                            v = get_surface(t, 1, x == nx)
                            offset += append_face!(pts, tri, cell_ix, t, v, offset)
                        end
                        if y == 1 || y == ny || include_all
                            v = get_surface(t, 2, y == ny)
                            offset += append_face!(pts, tri, cell_ix, t, v, offset)    
                        end
                        if z == 1 || z == nz || include_all
                            v = get_surface(t, 3, z == nz)
                            offset += append_face!(pts, tri, cell_ix, t, v, offset)    
                        end
                    end
                end
            end
        else
            for y in 1:ny
                for x in 1:nx
                    for z in 1:nz
                        t = (x, y, z)
                        for st in [true, false]
                            v = get_surface(t, 1, st)
                            offset += append_face!(pts, tri, cell_ix, t, v, offset)
                            v = get_surface(t, 2, st)
                            offset += append_face!(pts, tri, cell_ix, t, v, offset)
                            v = get_surface(t, 3, st)
                            offset += append_face!(pts, tri, cell_ix, t, v, offset)
                        end
                    end
                end
            end
        end
    end
    pts = plot_flatten_helper(pts)
    tri = plot_flatten_helper(tri)

    cell_ix = vcat(cell_ix...)
    face_index = vcat(face_index...)

    mapper = (
                Cells = (cell_data) -> cell_data[cell_ix],
                Faces = (face_data) -> face_data[face_index],
                indices = (Cells = cell_ix, Faces = face_index)
              )
    return (points = pts, triangulation = tri, mapper = mapper)
end
