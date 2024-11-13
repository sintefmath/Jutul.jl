function trajectory_to_points(trajectory::Matrix{Float64})
    N = size(trajectory, 2)
    @assert N in (2, 3) "2D/3D matrices are supported."
    return collect(vec(reinterpret(SVector{N, Float64}, collect(trajectory'))))
end

function trajectory_to_points(x::AbstractVector{SVector{N, Float64}}) where N
    return x
end


"""
    find_enclosing_cells(G, traj; geometry = tpfv_geometry(G), n = 25)

Find the cell indices of cells in the mesh `G` that are intersected by a given
trajectory `traj`. `traj` can be either a matrix with equal number of columns as
dimensions in G (i.e. three columns for 3D) or a `Vector` of `SVector` instances
with the same length.

The optional argument `geometry` is used to define the centroids and normals
used in the calculations. You can precompute this if you need to perform many
searches. The keyword argument `n` can be used to set the number of
discretizations in each segment.

`use_boundary` is by default set to `false`. If set to true, the boundary faces
of cells are treated more rigorously when picking exactly what cells are cut by
a trajectory, but this requires that the boundary normals are oriented outwards,
which is currently not the case for all meshes from downstream packages.

Examples:
```
# 3D mesh
G = CartesianMesh((4, 4, 5), (100.0, 100.0, 100.0))
trajectory = [
    50.0 25.0 1;
    55 35.0 25;
    65.0 40.0 50.0;
    70.0 70.0 90.0
]

cells = Jutul.find_enclosing_cells(G, trajectory)

# Optional plotting, requires Makie:
fig, ax, plt = Jutul.plot_mesh_edges(G)
plot_mesh!(ax, G, cells = cells, alpha = 0.5, transparency = true)
lines!(ax, trajectory, linewidth = 10)
fig
# 2D mesh
G = CartesianMesh((50, 50), (1.0, 2.0))
trajectory = [
    0.1 0.1;
    0.2 0.4;
    0.3 1.2
]
fig, ax, plt = Jutul.plot_mesh_edges(G)
cells = Jutul.find_enclosing_cells(G, trajectory)
# Plotting, needs Makie
fig, ax, plt = Jutul.plot_mesh_edges(G)
plot_mesh!(ax, G, cells = cells, alpha = 0.5, transparency = true)
lines!(ax, trajectory[:, 1], trajectory[:, 2], linewidth = 3)
fig
```
"""
function find_enclosing_cells(G, traj; geometry = missing, n = 25, use_boundary = false)
    G = UnstructuredMesh(G)
    if ismissing(geometry)
        geometry = tpfv_geometry(G)
    end
    pts = trajectory_to_points(traj)
    T = eltype(pts)
    normals = vec(reinterpret(T, geometry.normals))
    face_centroids = vec(reinterpret(T, geometry.face_centroids))
    cell_centroids = vec(reinterpret(T, geometry.cell_centroids))

    boundary_centroids = vec(reinterpret(T, geometry.boundary_centroids))
    if use_boundary
        boundary_normals = vec(reinterpret(T, geometry.boundary_normals))
    else
        boundary_normals = boundary_centroids .- cell_centroids[G.boundary_faces.neighbors]
    end

    # Start search near middle of trajectory
    mean_pt = mean(pts)
    nc = number_of_cells(G)
    cells_by_dist = sort(1:nc, by = cell -> norm(cell_centroids[cell] - mean_pt, 2))

    nseg = length(pts)-1
    intersected_cells = Int[]
    lengths = Float64
    for i in 1:nseg
        pt_start = pts[i]
        pt_end = pts[i+1]
        for pt in range(pt_start, pt_end, n)
            ix = find_enclosing_cell(G, pt, normals, face_centroids, boundary_normals, boundary_centroids, cells_by_dist)
            if !isnothing(ix)
                push!(intersected_cells, ix)
            end
        end
    end
    return unique!(intersected_cells)
end

"""
    find_enclosing_cell(G::UnstructuredMesh{D}, pt::SVector{D, T},
        normals::AbstractVector{SVector{D, T}},
        face_centroids::AbstractVector{SVector{D, T}},
        boundary_normals::AbstractVector{SVector{D, T}},
        boundary_centroids::AbstractVector{SVector{D, T}},
        cells = 1:number_of_cells(G)
    ) where {D, T}

Find enclosing cell of a point. This can be a bit expensive for larger meshes.
Recommended to use the more high level `find_enclosing_cells` instead.
"""
function find_enclosing_cell(G::UnstructuredMesh{D}, pt::SVector{D, T},
        normals::AbstractVector{SVector{D, T}},
        face_centroids::AbstractVector{SVector{D, T}},
        boundary_normals::AbstractVector{SVector{D, T}},
        boundary_centroids::AbstractVector{SVector{D, T}},
        cells = 1:number_of_cells(G)
    ) where {D, T}
    inside_normal(pt, normal, centroid) = dot(normal, pt - centroid) <= 0
    for cell in cells
        inside = true
        for face in G.faces.cells_to_faces[cell]
            if G.faces.neighbors[face][1] == cell
                sgn = 1
            else
                sgn = -1
            end
            normal = sgn*normals[face]
            center = face_centroids[face]
            inside = inside && inside_normal(pt, normal, center)
            if !inside
                break
            end
        end
        if !inside
            continue
        end

        for bface in G.boundary_faces.cells_to_faces[cell]
            normal = boundary_normals[bface]
            center = boundary_centroids[bface]
            inside = inside && inside_normal(pt, normal, center)
            if !inside
                break
            end
        end
        if inside
            return cell
        end
    end
    return nothing
end
