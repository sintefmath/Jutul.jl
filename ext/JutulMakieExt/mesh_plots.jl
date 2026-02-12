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
        keep_cells = Dict{Int, Bool}()
        keep_faces = Dict{Int, Bool}()
        keep_bf = Dict{Int, Bool}()

        if eltype(cells) == Bool
            @assert length(cells) == number_of_cells(m)
            cells = findall(cells)
        end
        if has_cell_filter
            for c in cells
                keep_cells[c] = true
            end
        end
        if eltype(faces) == Bool
            @assert length(faces) == number_of_faces(m)
            faces = findall(faces)
        end
        if has_face_filter
            for f in faces
                keep_faces[f] = true
            end
        end
        if eltype(boundaryfaces) == Bool
            @assert length(boundaryfaces) == number_of_boundary_faces(m)
            boundaryfaces = findall(boundaryfaces)
        end
        if has_bface_filter
            nf = number_of_faces(m)
            boundaryfaces = deepcopy(boundaryfaces)
            boundaryfaces .+= nf
            for f in boundaryfaces
                keep_bf[f] = true
            end
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
                keep_this = keep_this && haskey(keep_cells, cell_ix[tri_tmp])
            end
            if has_face_filter
                keep_this = keep_this && haskey(keep_faces, face_ix[tri_tmp])
            end
            if has_bface_filter
                keep_this = keep_this && haskey(keep_bf, face_ix[tri_tmp])
            end
            keep[i] = keep_this
        end
        tri = tri[keep, :]
        tri, pts, = remove_unused_points(tri, pts)
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
    return (tri, pts, unique_pts_ix)
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
    keep = map(isfinite, color)

    keep_tri = Int[]
    for i in axes(tri, 1)
        keep = true
        for j in axes(tri, 2)
            keep = keep && isfinite(color[tri[i, j]])
        end
        if keep
            push!(keep_tri, i)
        end
    end
    tri = tri[keep_tri, :]
    tri, pts, unique_pts_ix = remove_unused_points(tri, pts)
    if color isa AbstractVector
        color = color[unique_pts_ix]
    elseif color isa AbstractMatrix
        color = color[unique_pts_ix, :]
    end
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
        faces = nothing,
        boundary_faces = nothing,
        outer = dim(m) == 3 && isnothing(faces) && isnothing(boundary_faces),
        linewidth = 0.3,
        kwarg...
    )
    m = physical_representation(m)
    s = Jutul.mesh_linesegments(m, cells = cells, faces = faces, boundary_faces = boundary_faces, outer = outer)
    f = linesegments!(ax, s; linewidth = linewidth, transparency = transparency, color = color, kwarg...)
    return f
end
