function Jutul.plot_mesh_impl(m; kwarg...)
    fig, ax = basic_3d_figure()
    p = Jutul.plot_mesh!(ax, m; kwarg...)
    display(fig)
    return (fig, ax, p)
end

function Jutul.plot_mesh_impl!(ax, m; color = :lightblue, kwarg...)
    pts, tri, mapper = triangulate_mesh(m)
    f = mesh!(ax, pts, tri; color = color, kwarg...)
    return f
end

function Jutul.plot_cell_data_impl(m, data; colorbar = :vertical, kwarg...)
    fig, ax = basic_3d_figure()
    p = Jutul.plot_cell_data!(ax, m, data; kwarg...)
    if !isnothing(colorbar) && maximum(data) != minimum(data)
        if colorbar == :vertical
            Colorbar(fig[2, 1], p, vertical = false)
        else
            Colorbar(fig[1, 2], p, vertical = true)
        end
    end
    display(fig)
    return (fig, ax, p)
end

function Jutul.plot_cell_data_impl!(ax, m, data::AbstractVecOrMat, kwarg...)
    pts, tri, mapper = triangulate_mesh(m)
    return mesh!(ax, pts, tri; color = mapper.Cells(data), kwarg...)
end
