struct CartesianMesh <: AbstractTervMesh
    dims   # Tuple of dimensions (x, y, [z])
    deltas # Either a tuple of scalars (uniform grid) or a tuple of vectors (non-uniform grid)
    origin # Coordinate of lower left corner
    function CartesianMesh(dims::Tuple, deltas_or_size::Union{Nothing, Tuple} = nothing; origin = nothing)
        dim = length(dims)
        if isnothing(deltas_or_size)
            deltas_or_size = Tuple(ones(dim))
        end
        if isnothing(origin)
            origin = zeros(dim)
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
        return new(dims, deltas, origin)
    end
end

dim(t::CartesianMesh) = length(t.dims)

function tpfv_geometry(g::CartesianMesh)
    Δ = g.deltas
    d = dim(g)

    nx, ny, nz = get_3d_dims(g)

    cell_index(x, y, z) = (z-1)*nx*ny + (y-1)*nx + x
    get_deltas(x, y, z) = (get_delta(Δ, x, 1), get_delta(Δ, y, 2), get_delta(Δ, z, 3))
    # Cell data first - volumes and centroids
    nc = nx*ny*nz
    
    function cell_center(pos, i, δ)
        if isa(δ, AbstractFloat)
            return (pos[i] - 0.5)*δ
        else
            return sum(δ[1:(i-1)]) + δ[i]/2
        end
    end
    V = zeros(nc)
    cell_centroids = zeros(d, nc)
    for x in 1:nx
        for y in 1:ny
            for z = 1:nz
                pos = (x, y, z)
                c = cell_index(pos...)
                Δx, Δy, Δz  = get_deltas(pos...)
                V[c] = Δx*Δy*Δz

                for i in 1:d
                    cell_centroids[i, c] = cell_center(pos, i, Δ[i]) + g.origin[i]
                end
            end
        end
    end

    # Then face data:
    nf = (nx-1)*ny*nz + (ny-1)*nx*nz + (nz-1)*ny*nx
    N = Matrix{Int}(undef, 2, nf)
    face_areas = Vector{Float64}(undef, nf)
    face_centroids = zeros(d, nf)
    face_normals = zeros(d, nf)

    function add_face!(face_areas, face_normals, face_centroids, x, y, z, D, pos)
        isX = D == 1
        isY = D == 2
        isZ = D == 3
        @info isX
        index = cell_index(x, y, z)
        N[1, pos] = index
        N[2, pos] = cell_index(x + isX, y + isY, z + isZ)

        Δx, Δy, Δz  = get_deltas(x, y, z)

        face_areas[pos] = (Δx*!isX)*(Δy*!isY)*(Δz*!isZ)
        face_normals[D, pos] = 1.0

        face_centroids[:, pos] = cell_centroids[:, index]
        # Offset by the grid size
        face_centroids[D, pos] += Δy/2.0
    end
    # Note: The following loops are arranged to reproduce the MRST ordering.
    pos = 1
    # Faces with X-normal > 0
    for z = 1:nz
        for y in 1:ny
            for x in 1:(nx-1)
                add_face!(face_areas, face_normals, face_centroids, x, y, z, 1, pos)
                pos += 1
                @info N[:, pos-1]
            end
        end
    end
    # Faces with Y-normal > 0
    for y in 1:(ny-1)
        for z = 1:nz
            for x in 1:nx
                add_face!(face_areas, face_normals, face_centroids, x, y, z, 2, pos)
                pos += 1
            end
        end
    end
    # Faces with Z-normal > 0
    for z = 1:(nz-1)
        for y in 1:ny
            for x in 1:nx
                add_face!(face_areas, face_normals, face_centroids, x, y, z, 3, pos)
                pos += 1
            end
        end
    end

    return TwoPointFiniteVolumeGeometry(N, face_areas, V, face_normals, cell_centroids, face_centroids)
end

function get_3d_dims(g)
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
