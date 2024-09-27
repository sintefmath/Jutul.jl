function Jutul.plot_mesh_impl(m;
        resolution = default_jutul_resolution(),
        z_is_depth = Jutul.mesh_z_is_depth(m),
        kwarg...
    )
    if dim(m) == 3
        makefig = basic_3d_figure
    else
        makefig = basic_2d_figure
    end
    fig, ax = makefig(resolution, z_is_depth = z_is_depth)
    p = Jutul.plot_mesh!(ax, m; kwarg...)
    display(fig)
    return (fig, ax, p)
end

function Jutul.plot_mesh_impl!(ax, m;
        cells = nothing,
        faces = nothing,
        boundaryfaces = nothing,
        outer = false,
        color = :lightblue,
        kwarg...
    )
    pts, tri, mapper = triangulate_mesh(m, outer = outer)
    has_cell_filter = !isnothing(cells)
    has_face_filter = !isnothing(faces)
    has_bface_filter = !isnothing(boundaryfaces)
    if has_cell_filter || has_face_filter || has_bface_filter
        if eltype(cells) == Bool
            @assert length(cells) == number_of_cells(m)
            cells = findall(cells)
        end
        if eltype(faces) == Bool
            @assert length(faces) == number_of_faces(m)
            faces = findall(faces)
        end
        if eltype(boundaryfaces) == Bool
            @assert length(boundaryfaces) == number_of_boundary_faces(m)
            boundaryfaces = findall(boundaryfaces)
        end
        if has_bface_filter
            nf = number_of_faces(m)
            boundaryfaces = deepcopy(boundaryfaces)
            boundaryfaces .+= nf
        end
        ntri = size(tri, 1)
        keep = fill(false, ntri)
        cell_ix = mapper.indices.Cells
        face_ix = mapper.indices.Faces
        for i in 1:ntri
            # All tris have same cell so this is ok
            tri_tmp = tri[i, 1]
            keep_this = true
            if has_cell_filter
                keep_this = keep_this && cell_ix[tri_tmp] in cells
            end
            if has_face_filter
                keep_this = keep_this && face_ix[tri_tmp] in faces
            end
            if has_bface_filter
                keep_this = keep_this && face_ix[tri_tmp] in boundaryfaces
            end
            keep[i] = keep_this
        end
        tri = tri[keep, :]
        tri, pts = remove_unused_points(tri, pts)
    end
    if length(pts) > 0
        f = mesh!(ax, pts, tri; color = color, backlight = 1, kwarg...)
    else
        f = nothing
    end
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
    if dim(m) == 3
        makefig = basic_3d_figure
    else
        makefig = basic_2d_figure
    end
    fig, ax = makefig(resolution, z_is_depth = z_is_depth)

    p = Jutul.plot_cell_data!(ax, m, data; kwarg...)
    min_data = minimum(data)
    max_data = maximum(data)
    if !isnothing(colorbar) && colorbar !=false && min_data != max_data
        # ticks = range(min_data, max_data, 10)
        if colorbar == :horizontal
            Colorbar(fig[2, 1], p, vertical = false)
        else
            @assert colorbar == :vertical
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
        outer = dim(m) == 3,
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
