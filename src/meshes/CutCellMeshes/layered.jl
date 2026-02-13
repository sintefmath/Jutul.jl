"""
    depth_grid_to_surface(xs, ys, depths)

Convert a regular depth grid to a triangulated [`PolygonalSurface`](@ref).

`xs` and `ys` are vectors of coordinates defining the grid axes.  `depths` is a
matrix of size `(length(xs), length(ys))` giving the depth (z-coordinate) at
each grid node.  The grid is triangulated into pairs of triangles per
rectangular cell.

Entries that are `NaN` or `missing` are skipped, so incomplete horizons are
supported.

The resulting surface normal points upward (positive z) for each triangle.
"""
function depth_grid_to_surface(
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    depths::AbstractMatrix{<:Real}
)
    T = Float64
    nx = length(xs)
    ny = length(ys)
    @assert size(depths) == (nx, ny) "depths must be (length(xs), length(ys))"

    polygons = Vector{SVector{3, T}}[]

    for i in 1:(nx - 1)
        for j in 1:(ny - 1)
            z00 = depths[i,     j    ]
            z10 = depths[i + 1, j    ]
            z01 = depths[i,     j + 1]
            z11 = depths[i + 1, j + 1]

            p00 = SVector{3, T}(xs[i],     ys[j],     z00)
            p10 = SVector{3, T}(xs[i + 1], ys[j],     z10)
            p01 = SVector{3, T}(xs[i],     ys[j + 1], z01)
            p11 = SVector{3, T}(xs[i + 1], ys[j + 1], z11)

            # Lower-left triangle: p00-p10-p01
            if _depth_valid(z00) && _depth_valid(z10) && _depth_valid(z01)
                push!(polygons, [p00, p10, p01])
            end
            # Upper-right triangle: p10-p11-p01
            if _depth_valid(z10) && _depth_valid(z11) && _depth_valid(z01)
                push!(polygons, [p10, p11, p01])
            end
        end
    end

    return PolygonalSurface(polygons)
end

_depth_valid(z::Real) = isfinite(z)
_depth_valid(::Missing) = false

"""
    layered_mesh(mesh, surfaces; min_cut_fraction=0.01)

Build a layered reservoir mesh from a base mesh and a set of depth surfaces.

`mesh` is a 3D `UnstructuredMesh`.  `surfaces` is a vector of
[`PolygonalSurface`](@ref) objects ordered by increasing depth (shallowest
first).  Each surface defines a horizon that separates two layers.

The function returns `(result_mesh, info)` where:
- `result_mesh` is an `UnstructuredMesh` with the same total volume as the
  original mesh (all cells are preserved, just split at surfaces).
- `info` is a `Dict{String, Any}` containing:
  - `"layer_indices"`: A `Vector{Int}` of length `number_of_cells(result_mesh)`.
    Each entry gives the layer number of the corresponding cell.  Layer `0` is
    above all surfaces, layer `k` (1 ≤ k ≤ N-1 where N = length(surfaces))
    is between surface `k` and surface `k+1`, and layer `N` is below all
    surfaces.
  - `"cell_index"`: A `Vector{Int}` mapping each new cell to the original
    cell index in the input mesh.

# Algorithm

Each surface is applied sequentially from shallowest to deepest using the
standard `cut_mesh` (keeping both halves of every cut cell).  After all cuts,
each cell's centroid is compared against every surface to determine its layer.

If a shallower surface dips below a deeper one at some location, the shallower
surface takes priority (the cell is classified by the topmost surface it is
below).
"""
function layered_mesh(
    mesh::UnstructuredMesh{3},
    surfaces::Vector{<:PolygonalSurface};
    min_cut_fraction::Real = 0.01
)
    T = Float64
    n_surfaces = length(surfaces)
    nc_orig = number_of_cells(mesh)

    # ----------------------------------------------------------------
    # 1. Apply all surface cuts sequentially
    # ----------------------------------------------------------------
    current_mesh = mesh
    cell_map = collect(1:nc_orig)  # maps new cell → original cell

    for (si, surface) in enumerate(surfaces)
        result, step_info = cut_mesh(current_mesh, surface;
            extra_out = true,
            min_cut_fraction = min_cut_fraction
        )

        # Compose cell maps
        cell_map = [cell_map[j] for j in step_info["cell_index"]]
        current_mesh = result
    end

    # ----------------------------------------------------------------
    # 2. Classify each cell by layer using centroid position
    # ----------------------------------------------------------------
    nc_final = number_of_cells(current_mesh)
    geo = tpfv_geometry(current_mesh)
    layer_indices = Vector{Int}(undef, nc_final)

    for c in 1:nc_final
        cx = geo.cell_centroids[1, c]
        cy = geo.cell_centroids[2, c]
        cz = geo.cell_centroids[3, c]
        centroid = SVector{3, T}(cx, cy, cz)

        # Determine layer by counting how many surfaces the centroid is below.
        # Surfaces are ordered by increasing depth, so once the centroid is
        # above a surface we can stop: it cannot be below any deeper surface
        # without contradicting the ordering requirement.
        layer = 0
        for (si, surface) in enumerate(surfaces)
            if _point_below_surface(centroid, surface)
                layer = si
            else
                break
            end
        end
        layer_indices[c] = layer
    end

    info = Dict{String, Any}(
        "layer_indices" => layer_indices,
        "cell_index" => cell_map
    )

    return (current_mesh, info)
end

"""
    _point_below_surface(pt, surface)

Check whether a 3D point is deeper than (below) a polygonal surface.

For each polygon the signed distance from the polygon's plane is computed.  The
polygon whose centroid is closest in the horizontal (xy) plane is used.  A
positive signed distance from that plane (in the direction of the polygon
normal) corresponds to *greater depth* when the normal is oriented in the
depth-increasing direction — which is the default produced by
[`depth_grid_to_surface`](@ref).
"""
function _point_below_surface(pt::SVector{3, T}, surface::PolygonalSurface) where T
    best_dist = T(Inf)
    best_sd = zero(T)

    for (i, poly) in enumerate(surface.polygons)
        c = sum(poly) / length(poly)
        dx = pt[1] - c[1]
        dy = pt[2] - c[2]
        xy_dist = dx^2 + dy^2

        if xy_dist < best_dist
            best_dist = xy_dist
            plane = PlaneCut(c, surface.normals[i])
            best_sd = signed_distance(plane, pt)
        end
    end

    return best_sd > 0
end
