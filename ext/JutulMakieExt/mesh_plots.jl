function Jutul.plot_mesh_impl(m;
        resolution = default_jutul_resolution(),
        z_is_depth = Jutul.mesh_z_is_depth(m),
        kwarg...
    )
    fig, ax = basic_3d_figure(resolution, z_is_depth = z_is_depth)
    p = Jutul.plot_mesh!(ax, m; kwarg...)
    display(fig)
    return (fig, ax, p)
end

function Jutul.plot_mesh_impl!(ax, m;
        cells = nothing,
        outer = false,
        color = :lightblue,
        kwarg...
    )
    pts, tri, mapper = triangulate_mesh(m, outer = outer)
    if !isnothing(cells)
        if eltype(cells) == Bool
            @assert length(cells) == number_of_cells(m)
            cells = findall(cells)
        end
        ntri = size(tri, 1)
        keep = [false for i in 1:ntri]
        cell_ix = mapper.indices.Cells
        for i in 1:ntri
            # All tris have same cell so this is ok
            keep[i] = cell_ix[tri[i, 1]] in cells
        end
        tri = tri[keep, :]
        tri, pts = remove_unused_points(tri, pts)
    end
    f = mesh!(ax, pts, tri; color = color, backlight = 1, kwarg...)
    return f
end

function remove_unused_points(tri, pts)
    unique_pts_ix = unique(vec(tri))
    renum = Dict{Int, Int}()
    for (i, ix) in enumerate(unique_pts_ix)
        renum[ix] = i
    end
    pts = pts[unique_pts_ix, :]
    for i in eachindex(tri)
        tri[i] = renum[tri[i]]
    end
    return (tri, pts)
end

function Jutul.plot_cell_data_impl(m, data;
        colorbar = :horizontal,
        resolution = default_jutul_resolution(),
        z_is_depth = Jutul.mesh_z_is_depth(m),
        kwarg...
    )
    fig, ax = basic_3d_figure(resolution, z_is_depth = z_is_depth)
    p = Jutul.plot_cell_data!(ax, m, data; kwarg...)
    min_data = minimum(data)
    max_data = maximum(data)
    if !isnothing(colorbar) && min_data != max_data
        # ticks = range(min_data, max_data, 10)
        if colorbar == :horizontal
            Colorbar(fig[2, 1], p, vertical = false)
        else
            Colorbar(fig[1, 2], p, vertical = true)
        end
    end
    display(fig)
    return (fig, ax, p)
end

function Jutul.plot_cell_data_impl!(ax, m, data::AbstractVecOrMat; cells = nothing, outer = false, kwarg...)
    nc = number_of_cells(m)
    pts, tri, mapper = triangulate_mesh(m, outer = outer)
    data = vec(data)
    if !isnothing(cells)
        if eltype(cells) == Bool
            @assert length(cells) == nc
            cells = findall(cells)
        end
        new_data = fill(NaN, nc)
        if length(data) == length(cells)
            for (i, j) in enumerate(cells)
                new_data[j] = data[i]
            end
        else
            @assert length(data) == nc
            for i in cells
                new_data[i] = data[i]
            end
        end
        data = new_data
    end
    @assert length(data) == nc
    color = mapper.Cells(data)
    return mesh!(ax, pts, tri; backlight = 1, color = color, kwarg...)
end

function Jutul.plot_mesh_edges_impl(m;
        resolution = default_jutul_resolution(),
        z_is_depth = Jutul.mesh_z_is_depth(m),
        kwarg...
    )
    fig, ax = basic_3d_figure(resolution, z_is_depth = z_is_depth)
    p = Jutul.plot_mesh_edges!(ax, m; kwarg...)
    display(fig)
    return (fig, ax, p)
end


function Jutul.plot_mesh_edges_impl!(ax, m;
        transparency = true,
        color = :black,
        cells = nothing,
        outer = true,
        linewidth = 0.3,
        kwarg...)
    m = physical_representation(m)
    if isnothing(cells)
        cells = 1:number_of_cells(m)
    end
    s = Jutul.mesh_linesegments(m, cells = cells, outer = outer)
    f = linesegments!(ax, s; linewidth = linewidth, transparency = transparency, color = color, kwarg...)
    return f
end
