function Jutul.plot_mesh_impl(m; resolution = default_jutul_resolution(), kwarg...)
    fig, ax = basic_3d_figure(resolution)
    p = Jutul.plot_mesh!(ax, m; kwarg...)
    display(fig)
    return (fig, ax, p)
end

function Jutul.plot_mesh_impl!(ax, m; cells = nothing, is_depth = true, outer = false, color = :lightblue, kwarg...)
    pts, tri, mapper = triangulate_mesh(m, outer = outer, is_depth = is_depth)
    if !isnothing(cells)
        ntri = size(tri, 1)
        keep = [false for i in 1:ntri]
        cell_ix = mapper.indices.Cells
        for i in 1:ntri
            # All tris have same cell so this is ok
            keep[i] = cell_ix[tri[i, 1]] in cells
        end
        tri = tri[keep, :]
    end
    f = mesh!(ax, pts, tri; color = color, kwarg...)
    return f
end

function Jutul.plot_cell_data_impl(m, data;
        colorbar = :horizontal,
        resolution = default_jutul_resolution(),
        kwarg...
    )
    fig, ax = basic_3d_figure(resolution)
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

function Jutul.plot_cell_data_impl!(ax, m, data::AbstractVecOrMat; cells = nothing, is_depth = true, outer = false, kwarg...)
    nc = number_of_cells(m)
    pts, tri, mapper = triangulate_mesh(m, outer = outer, is_depth = is_depth)
    data = vec(data)
    if !isnothing(cells)
        new_data = zeros(nc)
        @. new_data = NaN
        if length(data) == length(cells)
            new_data[cells] = data
        else
            for i in cells
                new_data[i] = data[i]
            end
        end
        data = new_data
    end
    @assert length(data) == nc
    return mesh!(ax, pts, tri; color = mapper.Cells(data), kwarg...)
end
