using Jutul.StaticArrays
using OrderedCollections

struct PlotExplorerOutput
    fig
    lscene
    right_grid
    add_menu
    add_toggle
end

Base.display(pe::PlotExplorerOutput) = display(pe.fig)

function Jutul.plot_explorer_impl(m::Union{JutulMesh, DataDomain}; static = missing, dynamic = missing, kwarg...)
    if ismissing(static)
        if m isa DataDomain
            static = convert_dict(m.data, number_of_cells(m))
            if haskey(m, :cell_centroids)
                cc = m[:cell_centroids]
                d = size(cc, 1)
                static["X"] = view(cc, 1, :)
                static["Y"] = view(cc, 2, :)
                if d > 2
                    static["Z"] = view(cc, 3, :)
                end
                if haskey(m, :volumes)
                    static["Volumes"] = m[:volumes]
                end
            end
        else
            static = OrderedDict{String, Vector}()
        end
    end
    return Jutul.plot_explorer_impl(m, static, dynamic; kwarg...)
end

function Jutul.plot_explorer_impl(m::Union{JutulMesh, DataDomain}, plot_data::AbstractDict, dynamic::Union{Missing, Vector} = missing; verbose = false, kwarg...)
    if verbose
        jutul_message("plot_explorer", "Triangulating mesh for plotting...")
    end
    t_static = @elapsed points, ttri, tri = mesh_as_static(m)
    if verbose
        jutul_message("plot_explorer", "Mesh triangulation complete in $(round(t_static, sigdigits=3)) seconds. Setting up plot...")
    end
    m = physical_representation(m)
    return Jutul.plot_explorer_impl(m, points, ttri, tri, plot_data, dynamic; verbose = verbose, kwarg...)
end

function convert_dict(d::AbstractDict, nc::Int)
    out = OrderedDict{String, Any}()
    for (k, v) in pairs(d)
        if v isa Tuple
            # Handle internal data domain keys
            v = first(v)
        end
        sk = String(k)
        if v isa Vector && length(v) == nc && eltype(v) <: Number
            out[sk] = v
        elseif v isa Matrix && size(v, 2) == nc
            for row in axes(v, 1)
                out["$sk row $row"] = view(v, row, :)
            end
        end
    end
    return out
end

function preset_colors(name::Symbol)
    background_color = missing
    hist_colormap = missing
    if name == :viridis_dark
        colormap = :viridis
        background_colormap = :linear_ternary_blue_0_44_c57_n256
        textcolor = :white
    elseif name == :gist_yarg
        colormap = :coolwarm
        background_colormap = :gist_yarg
        textcolor = :black
        background_color = :lightgray
    elseif name == :seaborn_icefire_gradient
        colormap = :seaborn_icefire_gradient
        background_colormap = :linear_ternary_blue_0_44_c57_n256
        textcolor = :white
    elseif name == :batlow
        colormap = :batlow
        background_colormap = :linear_ternary_blue_0_44_c57_n256
        textcolor = :white
    elseif name == :turbo
        colormap = :turbo
        background_colormap = :linear_ternary_blue_0_44_c57_n256
        background_color = :white
        textcolor = :black
    else
        error("Unknown preset: $name")
    end
    if ismissing(hist_colormap)
        hist_colormap = colormap
    end
    return (
        colormap = colormap,
        background_colormap = background_colormap,
        hist_colormap = hist_colormap,
        textcolor = textcolor,
        backgroundcolor = background_color
    )
end

function Jutul.plot_explorer_impl(m::JutulMesh, points, ttri, indices, static, dynamic_data;
        preset = :viridis_dark,
        textcolor = missing,
        background_colormap = missing,
        colormap = missing,
        hist_colormap = missing,
        nbins = 25,
        use_highclip = Sys.isapple(),
        backgroundcolor = missing,
        extra_static = true,
        zreversed = Jutul.mesh_z_is_depth(m),
        edges = false,
        camarg = NamedTuple(),
        show_axis = false,
        aspect = missing,
        plot_pause = 1.0/30.0,
        verbose = false
    )
    default_colors = preset_colors(preset)
    if ismissing(colormap)
        colormap = default_colors.colormap
    end
    if ismissing(background_colormap)
        background_colormap = default_colors.background_colormap
    end
    if ismissing(textcolor)
        textcolor = default_colors.textcolor
    end
    if ismissing(backgroundcolor)
        backgroundcolor = default_colors.backgroundcolor
    end
    if ismissing(hist_colormap)
        hist_colormap = default_colors.hist_colormap
    end
    HAS_DYNAMIC_DATA = !ismissing(dynamic_data)
    # Data conversion
    nc = number_of_cells(m)
    plot_data = convert_dict(static, nc)
    if extra_static || length(keys(plot_data)) == 0
        if !haskey(plot_data, "X")
            geo = missing
            try
                geo = tpfv_geometry(m)
            catch

            end
            if !ismissing(geo)
                cc = geo.cell_centroids
                plot_data["X"] = view(cc, 1, :)
                plot_data["Y"] = view(cc, 2, :)
                if size(cc, 1) > 2
                    plot_data["Z"] = view(cc, 3, :)
                end
                plot_data["Volumes"] = geo.volumes
            end
        end
        plot_data["Cell ID"] = 1:number_of_cells(m)
        if (m isa Jutul.UnstructuredMesh || m isa Jutul.CartesianMesh) && Jutul.grid_dims_ijk(m) != (nc, 1, 1)
            ijk = map(i -> Jutul.cell_ijk(m, i), 1:nc)
            plot_data["I"] = map(x -> x[1], ijk)
            plot_data["J"] = map(x -> x[2], ijk)
            plot_data["K"] = map(x -> x[3], ijk)
        end
    end
    if HAS_DYNAMIC_DATA
        dynamic_data = [convert_dict(d, nc) for d in dynamic_data]
    end
    background_colormap = to_colormap(background_colormap)
    if !ismissing(aspect)
        length(aspect) == 3 || error("Aspect ratio must be a tuple of three values (scale_x, scale_y, scale_z)")
    end
    use_gradient = ismissing(backgroundcolor)
    if use_gradient
        scene_arg = (clear = false,)
    else
        scene_arg = (clear = true, backgroundcolor = backgroundcolor)
    end
    static_lims, dynamic_lims = setup_limits(plot_data, dynamic_data)
    menu_alpha = 0.1
    main_color = textcolor
    # Tri_T = eltype(ttri)
    W = 2
    pl = PointLight(RGBf(W, W, W), Point3f(0, 0, 0))
    # al = AmbientLight(RGBf(0.2, 0.2, 0.2))
    dl = DirectionalLight(RGBf(1, 1, 1), Vec3f(-1, 0.5, 1))
    lights = [pl]
    # lights = [pl, al]
    lights = [dl, pl]

    N = 20
    cmap = :seaborn_icefire_gradient
    cmap = :seaborn_mako_gradient
    # cmap = dark_purple
    bgcmap = :linear_ternary_blue_0_44_c57_n256
    # bgcmap = black_teal
    # bgcmap = midnight_blue_512

    bgcmap = background_colormap
    cmap = colormap
    lights = []

    fig = Figure(size = (1600, 800), figure_padding = 0.0)
    lscene = LScene(fig[1:N, 1:N], scenekw = (clear = false, ), show_axis = show_axis)
    mesh_scene = Scene(lscene.scene, scenekw = scene_arg)

    left_grid_layout = GridLayout(fig[:, 2:5], 10, 5)

    right_grid_layout_outer = GridLayout(fig[2:N-2, N-4:N-1], 3, 1)
    right_grid_layout = GridLayout(right_grid_layout_outer[1:2, 1])
    idx_right_gl = 2
    # Middle box for stepping
    step_grid_layout = GridLayout(fig[N-3:N, 6:16])
    idx_stepgl = 1

    histrng = 3
    hist_grid_layout = GridLayout(right_grid_layout_outer[histrng, 1])
    ax_hist = Axis(hist_grid_layout[1, 1],
        titlecolor = main_color,
        ygridvisible = false,
        xgridvisible = false,
        backgroundcolor = RGBAf(0.0, 0.0, 0.0, 0.0),
        xtickcolor = main_color,
        ytickcolor = main_color,
        xticklabelcolor = main_color,
        yticklabelcolor = main_color,
        # ax_hist.xticklabelsize = 0
        # ax_hist.xticksvisible = false
        # ax_hist.yticksvisible = false
    )

    function add_toggle!(title, checked = true; type = :checkbox, kwarg...)
        Label(right_grid_layout[idx_right_gl, 2:5], title, halign = :left, color = main_color)
        if type == :checkbox
            tog = Checkbox(right_grid_layout[idx_right_gl, 1], checked = checked)
        elseif type == :toggle
            tog = Toggle(right_grid_layout[idx_right_gl, 1], active = checked)
        else
            error("Unknown toggle type: $type")
        end
        idx_right_gl +=1
        return tog
    end

    function add_menu!(options, title = ""; prepend = false)
        options = collect(options)
        if prepend
            labels = map(o -> "$o ($title)", options)
        else
            labels = options
        end
        new_menu = Menu(right_grid_layout[idx_right_gl, 1:5],
            options = zip(labels, options),
            selection_cell_color_inactive = RGBAf(1, 1, 1, menu_alpha),
            textcolor = main_color
        )
        idx_right_gl += 1
        return new_menu
    end

    if HAS_DYNAMIC_DATA
        dyn_keys = keys(first(dynamic_data))
        Nstep = length(dynamic_data)
        toggle_dyn = add_toggle!("Show dynamic values", true, type = :toggle)
        toggle_static_limits = add_toggle!("Static color range", true, type = :toggle)
        toggle_independent = add_toggle!("Split filters", false, type = :toggle)
        is_dynamic = toggle_dyn.active
        is_independent = toggle_independent.active
        is_global_limit = toggle_static_limits.active
    else
        dyn_keys = ["No dynamic data"]
        is_global_limit = Observable(false)
        is_independent = Observable(false)
    end


    menu_cell = add_menu!(keys(plot_data), "static", prepend = HAS_DYNAMIC_DATA)

    slider_static = IntervalSlider(right_grid_layout[idx_right_gl, 1:5], range = 0:0.01:1, horizontal = true, startvalues = (0.0, 1.0))
    idx_right_gl += 1

    sel = menu_cell.selection
    value_static = slider_static.interval

    if HAS_DYNAMIC_DATA
        menu_dyn = add_menu!(dyn_keys, "dynamic", prepend = HAS_DYNAMIC_DATA)

        slider_dynamic = IntervalSlider(right_grid_layout[idx_right_gl, 1:5], range = 0:0.01:1, horizontal = true, startvalues = (0.0, 1.0))
        idx_right_gl += 1

        sel_dyn = menu_dyn.selection
        value_dynamic = slider_dynamic.interval

        lpos_tstep = idx_stepgl
        idx_stepgl += 1
        step_slider = Slider(step_grid_layout[idx_stepgl, 1:5], range = 1:Nstep, startvalue = 1, horizontal = true)
        step_idx = step_slider.selected_index
        idx_stepgl += 1

        tlabel = @lift string("Dynamic data step: ", $step_idx, "/$Nstep")
        Label(step_grid_layout[lpos_tstep, 1:5], tlabel, halign = :left, color = main_color)
        wb = 100

        start = Button(step_grid_layout[idx_stepgl, 1], label = "Start", width = wb)
        prev = Button(step_grid_layout[idx_stepgl, 2], label = "Previous", width = wb)
        play = Button(step_grid_layout[idx_stepgl, 3], label = "Play", width = wb)
        next = Button(step_grid_layout[idx_stepgl, 4], label = "Next", width = wb)
        lasts = Button(step_grid_layout[idx_stepgl, 5], label = "Last", width = wb)
        on(next.clicks) do _
            step_idx[] = min(step_idx[] + 1, Nstep)
            notify(step_idx)
        end
        on(prev.clicks) do _
            step_idx[] = max(step_idx[] - 1, 1)
            notify(step_idx)
        end
        on(start.clicks) do _
            step_idx[] = 1
            notify(step_idx)
        end
        on(lasts.clicks) do _
            step_idx[] = Nstep
            notify(step_idx)
        end
        # Play button logic
        is_playing = Ref(false)
        function playback()
            start = step_idx.val
            if start == Nstep
                step_idx[] = 1
                start = 1
            end
            previndex = start
            for _ in start:Nstep
                current = step_idx.val
                newindex = current + 1
                if newindex > Nstep || previndex != newindex-1 || !is_playing[]
                    break
                end
                plot_time = @elapsed step_idx[] = newindex
                previndex = newindex
                sleep(max(0, plot_pause - plot_time))
            end
        end
        on(play.clicks) do _
            if is_playing[]
                is_playing[] = false
            else
                is_playing[] = true
                @async playback()
            end
        end
    else
        is_dynamic = Observable(false)
        step_idx = Observable(1)
        sel_dyn = Observable("No dynamic data")
        value_dynamic = Observable((0.0, 1.0))
    end
    # Toggle mesh lines
    toggle_edge = add_toggle!("Mesh lines", edges)
    # Toggle mesh itself
    toggle_mesh = add_toggle!("Mesh cells")

    # Toggle mesh itself
    # transparency_toggle = add_toggle!("Transparency", false)
    hist_toggle = add_toggle!("Histogram", true)
    symlog_toggle = add_toggle!("Symlog10", false)
    cell_to_vertex = indices.Cells
    cell_val_buffer = zeros(nc)
    cell_val_buffer_trunc = zeros(nc)

    T_f = Float32
    vertex_val_buffer = GLMakie.Buffer(zeros(T_f, length(cell_to_vertex)))
    vertex_values = zeros(T_f, length(cell_to_vertex))
    function update_cell_values(static_key::String, dyn_key::String, step_idx::Int, bounds_static, bounds_dynamic, is_dyn, is_indep, is_glob, to_symlog)
        # Doing two things:
        # - Return the values in val_buffer for histogram
        # - Update the vertex_val_buffer for the mesh plotting
        if ismissing(dynamic_data)
            dyn_values = missing
        else
            dyn_values = dynamic_data[step_idx][dyn_key]
        end
        static_values = plot_data[static_key]
        if is_dyn
            v = dyn_values
        else
            v = static_values
        end
        if to_symlog
            F = symlog10
        else
            F = x -> x
        end
        @. cell_val_buffer = F(v)
        @. cell_val_buffer_trunc = v

        bnd_static = get_limits(static_lims, dynamic_lims, static_key, dyn_key, false, step_idx, is_glob, to_symlog)
        bnd_dyn = get_limits(static_lims, dynamic_lims, static_key, dyn_key, true, step_idx, is_glob, to_symlog)
        map_to_face_buffer_with_truncation!(vertex_val_buffer, vertex_values, cell_val_buffer_trunc, cell_to_vertex, bnd_dyn, bnd_static, dyn_values, static_values, bounds_dynamic, bounds_static, is_dyn, is_indep, use_highclip, F, verbose)

        # if do_map
        #     # out = tri.mapper.Cells(val_buffer)
        #     out = face_val_buffer
        #     @. out = val_buffer[cell_to_vertex]
        # else
        #     out = val_buffer
        # end
        return cell_val_buffer
    end

    # active = Tri_T[]

    use_symlog = symlog_toggle.checked
    # cdata_face = @lift get_mesh_plot($sel, $sel_dyn, $step_idx, $value_static, $value_dynamic, $is_dynamic, $is_global_limit, $use_symlog, do_map = true)
    cdata_cells = @lift update_cell_values($sel, $sel_dyn, $step_idx, $value_static, $value_dynamic, $is_dynamic, $is_independent, $is_global_limit, $use_symlog)
    lims = @lift get_limits(static_lims, dynamic_lims, $sel, $sel_dyn, $is_dynamic, $step_idx, $is_global_limit, $use_symlog)
    if use_highclip
        mesh_arg = (highclip = :transparent, )
    else
        mesh_arg = NamedTuple()
    end
    @info "??" points ttri
    mplt = mesh!(lscene, points, ttri;
        colormap = cmap,
        color = vertex_val_buffer,
        visible = toggle_mesh.checked,
        colorrange = lims,
        mesh_arg...
    )
    plot_mesh_edges!(lscene, m, visible = toggle_edge.checked, color = main_color)
    # plot_faults!(lscene, m, colormap = cmap)


    Colorbar(hist_grid_layout[2, 1], mplt,
        vertical = false,
        ticklabelsize = 12,
        flipaxis = false,
        ticklabelcolor = main_color,
        tickcolor = main_color
    )

    bins = @lift range($lims[1], $lims[2], length = nbins+1)
    bin_centers = @lift [($lims[1] + bin_idx*($lims[2] - $lims[1])/(2*nbins)) for bin_idx in 1:nbins]

    hist!(ax_hist, cdata_cells,
        color = bin_centers,
        colorrange = lims,
        colormap = hist_colormap,
        bins = bins,
        visible = hist_toggle.checked
    )
    hidespines!(ax_hist)
    on(hist_toggle.checked) do checked
        ax_hist.xticksvisible[] = checked
        ax_hist.yticksvisible[] = checked
        ax_hist.xticklabelsvisible[] = checked
        ax_hist.yticklabelsvisible[] = checked
    end

    # Axis fixes
    on(menu_cell.selection) do s
        if HAS_DYNAMIC_DATA
            toggle_dyn.active[] = false
        end
        autolimits!(ax_hist)
    end
    if HAS_DYNAMIC_DATA
        on(menu_dyn.selection) do _
            toggle_dyn.active[] = true
            autolimits!(ax_hist)
        end
        on(toggle_dyn.active) do _
            autolimits!(ax_hist)
        end
    end
    on(symlog_toggle.checked) do _
        autolimits!(ax_hist)
    end

    if zreversed
        upvector = Vec3f(0, 0, -1)
    else
        upvector = Vec3f(0, 0, 1)
    end
    cam = Makie.cam3d!(mesh_scene; upvector = upvector, camarg...)
    if Jutul.mesh_z_is_depth(m)
        # cam.upvector[] = Vec3f(0, 0, -1)
    end
    # w, h = size(scene_outer)
    # nearplane = 0.1f0
    # farplane = 100f0
    # aspect = Float32(w / h)
    # cam.projection[] = Makie.perspectiveprojection(45f0, aspect, nearplane, farplane)

    function cell_for_click(node_idx)
        return indices.Cells[node_idx]
    end

    selected_cells = Observable{Vector{Int}}(Int[])
    cell_outline = []

    colors_clicks = Makie.wong_colors()
    max_clicked = length(colors_clicks)
    on(events(fig).mousebutton, priority = 2) do event
        is_left = event.button == Mouse.left
        is_right = event.button == Mouse.middle
        is_shift = ispressed(fig, Keyboard.left_shift) || ispressed(fig, Keyboard.right_shift)
        if (is_left || is_right) && event.action == Mouse.press
            plt, i = pick(fig)
            if is_right
                for plt in cell_outline
                    delete!(mesh_scene, plt)
                end
                empty!(cell_outline)
                selected_cells[] = Int[]
                return Consume(false)
            elseif plt == mplt
                cell = cell_for_click(i)
                if verbose
                    jutul_message("plot_explorer", "Clicked cell: $cell")
                end
                for plt in cell_outline
                    delete!(mesh_scene, plt)
                end
                empty!(cell_outline)
                if is_shift
                    push!(selected_cells.val, cell)
                    while length(selected_cells.val) > max_clicked
                        popfirst!(selected_cells.val)
                    end
                    cells = selected_cells.val
                    notify(selected_cells)
                else
                    cells = [cell]
                    selected_cells[] = cells
                end
                for (i, cell) in enumerate(cells)
                    pp = plot_mesh_edges!(mesh_scene, m, cells = [cell], color = colors_clicks[i], linewidth = 1.5, outer = false)
                    push!(cell_outline, pp)
                end
                return Consume(true)
            else
                return Consume(false)
            end
        end
        return Consume(false)
    end

    function format_clicked_cell(cells::Vector{Int}, static::String, dynamic::String, step_idx)
        if length(cells) == 0
            return ""
        elseif length(cells) == 1
            cell = only(cells)
            str = "Cell $cell"
            ijk = missing
            try
                ijk = cell_ijk(m, cell)
                str *= " IJK=[$(ijk[1]),$(ijk[2]),$(ijk[3])]"
            catch
            end
            value = plot_data[static][cell]
            str *= "\n$static = $(round(value, sigdigits=4))"

            if !ismissing(dynamic_data)
                dyn_val = dynamic_data[step_idx][dynamic][cell]
                str *= "\n$dynamic = $(round(dyn_val, sigdigits=4))"
            end
        else
            str = "Cells:\n" * join(cells, ",\n")
        end
        return str
    end
    clicklabel = @lift format_clicked_cell($selected_cells, $sel, $sel_dyn, $step_idx)

    al = missing
    side_axis = missing
    function sideplot_timeseries(selected_c::Vector{Int}, dyn_key::String, is_dyn::Bool)
        if !ismissing(al)
            delete!(al)
        end
        if !ismissing(side_axis)
            empty!(side_axis)
            delete!(side_axis)
            side_axis = missing
        end
        if is_dyn && length(selected_c) > 0 && !ismissing(dynamic_data)
            side_axis = Axis(left_grid_layout[7:9, 1:5],
                backgroundcolor = RGBAf(0.0, 0.0, 0.0, 0.0),
                xtickcolor = main_color,
                ytickcolor = main_color,
                xticklabelcolor = main_color,
                yticklabelcolor = main_color,
                ylabel = "$dyn_key",
                xlabel = "Step index",
                ygridvisible = false,
                xgridvisible = false,
                topspinevisible = false,
                rightspinevisible = false,
                xlabelpadding = 1.0,
                ylabelpadding = 1.0,
                bottomspinecolor = main_color,
                leftspinecolor = main_color,
                xlabelcolor = main_color,
                ylabelcolor = main_color
            )
            for (cno, c) in enumerate(selected_c)
                values = [dynamic_data[i][dyn_key][c] for i in 1:length(dynamic_data)]
                lines!(side_axis, values, color = colors_clicks[cno], overdraw = true, label = "$c")
            end
            al = axislegend(framevisible = false, backgroundcolor = RGBAf(0.0, 0.0, 0.0, 0.0), labelcolor = main_color)
        end
        return missing
    end

    _ = @lift sideplot_timeseries($selected_cells, $sel_dyn, $is_dynamic)
    Label(left_grid_layout[2, 1:5], clicklabel, halign = :left, color = main_color, justification = :left)

    if !ismissing(aspect)
        scale!(lscene.scene, aspect...)
    end

    if use_gradient
        bgscene = Scene(lscene.scene)
        bg_plt = missing
        function draw_bg()
            campixel!(bgscene)
            w, h = size(bgscene) # get the size of the scene in pixels
            # this draws a line at the scene window boundary
            bg = [sin(i/w) + cos(j/h) for i in 1:w, j in 1:h]
            bg_plt = image!(bgscene, [0.5*i/w + (1.0 - (j/h)^1.5) for i in 1:w, j in 1:h],
                colormap = bgcmap,
                colorrange = (minimum(bg), maximum(bg))
            )
            translation_factor_bg = -10000
            if !ismissing(aspect) && length(aspect) == 3
                translation_factor_bg /= aspect[end]
            end
            translate!(bg_plt, 0, 0, translation_factor_bg)
        end
        draw_bg()

        on(lscene.scene.events.window_area) do s
            delete!(bgscene, bg_plt)
            draw_bg()
        end
    end
    return PlotExplorerOutput(fig, lscene, right_grid_layout, add_menu!, add_toggle!)
end

function mesh_as_static(m)
    tri = Jutul.triangulate_mesh(m, outer = false)
    return mesh_as_static(m, tri)
end

function mesh_as_static(m, tri)
    D = Jutul.dim(m)
    T = Jutul.float_type(m)
    return mesh_as_static(physical_representation(m), tri, Val(D), Val(T))
end

function convert_points(tripoints, ::Val{D}, ::Val{T}) where {D, T}
    Point_T = Point{D, T}
    points = Vector{Point_T}()
    npts = size(tripoints, 1)
    sizehint!(points, npts)
    if D == 2
        for i in 1:npts
            x = tripoints[i, 1]
            y = tripoints[i, 2]
            nextpt = Point_T((x, y))
            nextpt::Point_T
            push!(points, nextpt)
        end
    else
        @assert D == 3
        for i in 1:npts
            x = tripoints[i, 1]
            y = tripoints[i, 2]
            z = tripoints[i, 3]
            nextpt = Point_T((x, y, z))
            nextpt::Point_T
            push!(points, nextpt)
        end
    end
    return points
end

function mesh_as_static(m, tri, Dval::Val{D}, Tval::Val{T}) where {D, T}
    indices = tri.mapper.indices
    tripoints = tri.points
    triangulation = tri.triangulation
    tripoints::Matrix{T}
    points = convert_points(tripoints, Dval, Tval)
    Tri_T = Makie.GeometryBasics.TriangleFace{Int}
    ttri = Tri_T[]
    sizehint!(ttri, size(triangulation, 1))
    for i in axes(triangulation, 1)
        v1 = triangulation[i, 1]
        v2 = triangulation[i, 2]
        v3 = triangulation[i, 3]
        # Note the reversed order to get correct orientation for Makie?
        nexttri = Tri_T(v3, v2, v1)
        nexttri::Tri_T
        push!(ttri, nexttri)
    end
    return (points, ttri, indices)
end

function setup_limits(static, dynamic)
    static_lims = Dict()
    for (k, v) in static
        static_lims[k] = extrema(v)
    end
    if ismissing(dynamic)
        dynamic_lims = missing
    else
        dynamic_lims = Dict()
        dyn_keys = collect(keys(first(dynamic)))
        for k in dyn_keys
            dynamic_lims[k] = map(i -> extrema(dynamic[i][k]), eachindex(dynamic))
        end
        for (k, v) in dynamic_lims
            static_lims[k] = (minimum(map(first, v)), maximum(map(last, v)))
        end
    end
    return (static_lims, dynamic_lims)
end

function get_limits(static, dynamic, key_static, key_dynamic, is_dynamic, step, is_global_limit, to_symlog)
    if is_dynamic
        key = key_dynamic
    else
        key = key_static
    end
    if is_dynamic && !is_global_limit
        if ismissing(dynamic)
            lims = missing
        else
            lims = dynamic[key][step]
        end
    else
        lims = get(static, key, missing)
    end
    if !ismissing(lims)
        # Make sure limits are not identical
        ϵ = 1e-3
        low_delta = max(lims[1] + ϵ, lims[1]*(1+ϵ))
        lims = (lims[1], max(low_delta, lims[2]))
        if to_symlog
            lims = (symlog10(lims[1]), symlog10(lims[2]))
        end
    end
    return lims
end

function map_to_face_buffer_with_truncation!(vertex_val_buffer, vertex_vals, cell_vals, cell_to_vertex, bnd_dyn, bnd_static, dyn_values, static_values, limiter_dynamic, limiter_static, is_dyn, is_indep, use_highclip, F, verbose)
    # map_to_face_buffer_with_truncation!(face_val_buffer, cell_to_vertex
    # val_buffer, bnd_dyn, bnd_static, dyn_values, static_values, bounds_dynamic, bounds_static, is_dyn, use_highclip
    ϵ = 1e-6
    has_dynamic = !ismissing(dyn_values)
    if ismissing(bnd_dyn)
        @assert !has_dynamic
        bnd_dyn = (0.0, 1.0)
    else
        bnd_dyn = (bnd_dyn[1], max(bnd_dyn[2], bnd_dyn[1] + ϵ))
    end
    bnd_static = (bnd_static[1], max(bnd_static[2], bnd_static[1] + ϵ))
    is_outside(x, rng) = x < rng[1] || x > rng[2]
    to_inner(x, bnds) = (F(x) - bnds[1])/(bnds[2] - bnds[1])
    for cell_no in eachindex(cell_vals)
        if has_dynamic
            dyn_norm = to_inner(dyn_values[cell_no], bnd_dyn)
        else
            dyn_norm = 0.5
        end
        static_norm = to_inner(static_values[cell_no], bnd_static)
        outside_dyn = is_outside(dyn_norm, limiter_dynamic)
        outside_static = is_outside(static_norm, limiter_static)
        if is_indep
            skip = outside_dyn && outside_static
        else
            skip = outside_dyn || outside_static
        end
        if skip
            if use_highclip
                if is_dyn
                    cell_vals[cell_no] = (1.0 + ϵ)*bnd_dyn[2] + ϵ
                else
                    cell_vals[cell_no] = (1.0 + ϵ)*bnd_static[2] + ϵ
                end
            else
                cell_vals[cell_no] = NaN
            end
        end
    end
    @. vertex_vals = cell_vals[cell_to_vertex]
    n = length(vertex_vals)
    t_update = @elapsed vertex_val_buffer[1:n] = vertex_vals
    if verbose
        jutul_message("plot_explorer", "Updated vertex buffer in $(round(t_update, sigdigits=3)) seconds")
    end
    return vertex_val_buffer
end

function symlog10(x)
    # Inspired by matplotlib.scale.SymmetricalLogScale
    # https://matplotlib.org/stable/api/scale_api.html#matplotlib.scale.SymmetricalLogScale
    if x < 1.0 && x > -1.0
        transformed_val = x
    else
        transformed_val = sign(x)*(log10(abs(x))+1)
    end
    return transformed_val
end
