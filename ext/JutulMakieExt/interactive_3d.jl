import Jutul: plot_interactive_impl


function plot_interactive_impl(d::DataDomain, data = missing; kwarg...)
    mesh = physical_representation(d)
    if ismissing(data)
        plot_d = Dict{Symbol, Any}()
        for (k, v) in d.data
            val, e = v
            plot_d[k] = val
        end
        data = [plot_d]
    end
    plot_interactive_impl(mesh, data; kwarg...)
end

function plot_interactive_impl(model::MultiModel, states, model_key = nothing; kwarg...)
    if states isa AbstractDict
        states = [states]
    end
    if isnothing(model_key)
        model_key = first(keys(model.models))
    end
    if haskey(states[1], model_key)
        model_states = map(x -> x[model_key], states)
    else
        # Hope that the user knew what they sent in
        model_states = states
    end
    return plot_interactive_impl(model[model_key], model_states; kwarg...)
end


function plot_interactive_impl(model::SimulationModel, states; kwarg...)
    if states isa AbstractDict
        states = [states]
    end
    mesh = physical_representation(model.data_domain)
    if isnothing(mesh)
        @warn "No plotting possible. SimulationModel has .data_domain = nothing." 
    else
        return plot_interactive_impl(mesh, states; kwarg...)
    end
end

default_jutul_resolution() = (1600, 900)

function plot_interactive_impl(grid, states;
        plot_type = nothing,
        primitives = nothing,
        transparency = false,
        resolution = default_jutul_resolution(),
        alpha = 1.0,
        title = "",
        transform = "none",
        free_cam = false,
        new_window = get(ENV, "CI", "false") == "false",
        edge_color = nothing,
        step = 1,
        row = 1,
        key = missing,
        edge_arg = NamedTuple(),
        aspect = (1.0, 1.0, 1/3),
        colormap = :viridis,
        alphamap = :no_alpha_map,
        z_is_depth = missing,
        kwarg...
    )
    if ismissing(z_is_depth)
        if grid isa Integer
            z_is_depth = false
        else
            z_is_depth = Jutul.mesh_z_is_depth(grid)
        end
    end
    has_primitives = !isnothing(primitives)
    active_filters = []
    if states isa AbstractDict || states isa DataDomain
        states = [states]
    end
    if states isa AbstractVecOrMat && eltype(states)<:AbstractFloat
        states = [Dict(:data => states)]
    end
    if grid isa Integer
        # Assume that someone figured out primitives already...
        nc = grid
        @assert has_primitives
        if isnothing(plot_type)
            plot_type = :mesh
        end
    else
        nc = number_of_cells(grid)
    end
    current_filter = collect(false for i in 1:nc)
    if !has_primitives
        if isnothing(plot_type)
            plot_candidates = [:mesh, :meshscatter, :lines]
            for p in plot_candidates
                primitives = plot_primitives(grid, p)
                if !isnothing(primitives)
                    plot_type = p
                    break
                end
            end
            if isnothing(primitives)
                @warn "No suitable plot found for mesh of type $(typeof(grid)). I tried $plot_candidates"
                return
            end
        else
            primitives = plot_primitives(grid, plot_type)
            if isnothing(primitives)
                @warn "Mesh of type $(typeof(grid)) does not support plot_type :$plot_type"
                return
            end
        end
    end
    pts = primitives.points
    mapper = primitives.mapper

    fig = Figure(size = resolution)
    if states isa AbstractDict
        states = [states]
    end
    if eltype(states)<:Number && (length(states) == nc || size(states, 1) == nc)
        states = [Dict(:Data => states)]
    end
    data = states[1]
    labels = Vector{String}()
    row_limits = Dict()
    limits = Dict()
    for k in keys(data)
        d = data[k]
        is_valid_vec = d isa AbstractVector && length(d) == nc
        is_valid_mat = d isa AbstractMatrix && size(d, 2) == nc
        if eltype(d)<:Real && (is_valid_vec || is_valid_mat) 
            push!(labels, "$k")
            mv = Inf
            Mv = -Inf
            ismat = states[1][k] isa Matrix

            for s in states
                di = s[k]
                mv, Mv = my_minmax(di, mv, Mv)
            end
            if mv == Mv
                Mv = 1.01*mv + 1e-12
            end
            limits["$k"] = (mv, Mv)
            row_limits["$k"] = Dict{Int, Tuple}()
            if ismat
                nrows = size(states[1][k], 1)
                if nrows == 1
                    row_limits["$k"][1] = (mv, Mv)
                else
                    for row in 1:nrows
                        mv = Inf
                        Mv = -Inf
                        for s in states
                            di = view(s[k], row, :)
                            mv, Mv = my_minmax(di, mv, Mv)
                        end
                        row_limits["$k"][row] = (mv, Mv)
                    end
                end
            else
                row_limits["$k"][1] = (mv, Mv)
            end
        else
            @debug "Skipping $k: Non-numeric type" eltype(d)
        end
    end
    function get_valid_rows(s)
        sample = data[Symbol(s)]
        if sample isa AbstractVector
            n = 1
        else
            n = size(sample, 1)
        end
        return ["$x" for x in 1:n]
    end
    datakeys = labels
    if length(datakeys) == 0
        error("No plottable properties found.")
    end
    nstates = length(states)

    if ismissing(key)
        key = datakeys[1]
    elseif key isa Symbol
        key = "$key"
    end
    state_index = Observable{Int64}(step)
    function generate_title(base, ix)
        if length(base) == 0
            return "Step $(ix[])/$nstates"
        else
            return "$base step $(ix[])/$nstates"
        end
    end
    fig_title = Observable{String}(generate_title(title, state_index))
    row_index = Observable{Int64}(row)
    prop_name = Observable{Any}(key)
    transform_name = Observable{String}(transform)
    lims = Observable(limits[key])
    menu = Menu(fig, options = datakeys, default = key)
    menu_2 = Menu(fig, options = get_valid_rows("$key"), default = "1", width = 60)

    function change_index(ix; update_slider = true)
        tmp = clamp(ix, 1, nstates)
        if update_slider
            sl_x.selected_index = tmp
        end
        state_index[] = tmp
        fig_title[] = generate_title(title, state_index)
        lims[] = get_limits(menu_cscale.selection[], state_index.val, prop_name.val, row_index.val, states, transform_name[])
        return tmp
    end

    function increment_index(inc = 1)
        change_index(state_index.val + inc)
    end

    fig[5, 3] = hgrid!(
        menu,
        menu_2,
        ; tellheight = false, width = 300)

    sl_x = Slider(fig[5, 2], range = 1:nstates, value = state_index, snap = true)

    low = Observable{Float64}(0.0)
    hi = Observable{Float64}(1.0)

    rs_v = IntervalSlider(fig[4, :], range = LinRange(0, 1, 1000))

    on(rs_v.interval) do x
        low.val = x[1]
        hi.val = x[2]
        notify(prop_name)
    end
    # point = sl_x.value
    on(sl_x.selected_index) do n
        val = sl_x.selected_index.val
        change_index(val, update_slider = false)
    end
    is_3d = size(pts, 2) == 3
    ax_pos = fig[2, 1:3]
    if is_3d
        ax = Axis3(ax_pos, title = fig_title, aspect = aspect, zreversed = z_is_depth)
        if free_cam
            Camera3D(ax.scene)
        end
    else
        ax = Axis(ax_pos, title = fig_title)
    end
    cell_buffer = zeros(nc)
    # Selection of data
    ys = @lift(
                mapper.Cells(
                    select_data(
                        cell_buffer,
                        current_filter,
                        states[$state_index],
                        Symbol($prop_name),
                        $row_index,
                        $low,
                        $hi,
                        $lims,
                        $transform_name,
                        active_filters
                        )
                )::Vector{Float64}
            )
    # Selection of colormap
    colormap_name = Observable(colormap)
    alphamap_name = Observable(alphamap)
    cmap = @lift(generate_colormap($colormap_name, $alphamap_name, alpha, $low, $hi))

    # Menu for field to plot
    on(menu.selection) do s
        rows = get_valid_rows(s)
        msel = menu_2.selection[]
        if isnothing(msel)
            old = 1
        else
            old = parse(Int64, msel)
        end
        nextn = min(old, length(rows))
        prop_name.val = s
        row_index.val = nextn
        # notify(prop_name)
        # notify(menu_2.selection)
        menu_2.options = rows
        menu_2.selection.val = "$nextn"
        lims[] = get_limits(menu_cscale.selection[], state_index.val, s, nextn, states, transform_name[])
        notify(prop_name)
        # notify(menu_2.selection)
    end
    # Row of dataset selector
    on(menu_2.selection) do s
        if isnothing(s)
            s = "1"
        end
        nextn = parse(Int64, s)
        row_index[] = nextn
        lims[] = get_limits(menu_cscale.selection[], state_index.val, prop_name.val, nextn, states, transform_name[])
    end
    # Top row
    fig[1, :] = top_layout = GridLayout(2, 1, tellwidth = false)
    N_top = 1

    # Alpha map selector
    genlabel(l) = Label(fig, l, font = :bold)
    top_layout[1, N_top] = genlabel("Alphamap")
    N_top += 1

    alphamaps = ["no_alpha_map", "linear", "linear_scaled", "inv_linear", "inv_linear_scaled"]
    amap_str = "$alphamap"
    if !(amap_str in alphamaps)
        push!(alphamaps, cmap_str)
    end
    top_layout[1, N_top] = lmap = GridLayout()
    menu_amap = Menu(lmap[1, 1], options = alphamaps, prompt = amap_str)
    on(menu_amap.selection) do s
        alphamap_name[] = Symbol(s)
    end
    N_top += 1

    top_layout[1, N_top] = top_buttons = GridLayout(tellwidth = true)
    N_top += 1
    # Clear all filters
    function reset_selection_slider!()
        low[] = 0.0
        hi[] = 1.0
        set_close_to!(rs_v, 0.0, 1.0)
    end

    b_clear = Button(fig, label = "Clear all")
    on(b_clear.clicks) do _
        empty!(active_filters)
        notify(state_index)
    end
    b_clear_last = Button(fig, label = "Remove last")
    on(b_clear_last.clicks) do _
        if length(active_filters) > 0
            pop!(active_filters)
            notify(state_index)
        end
    end
    b_add_static = Button(fig, label = "Add static")
    on(b_add_static.clicks) do _
        if any(current_filter)
            push!(active_filters, copy(current_filter))
        end
        reset_selection_slider!()
    end
    b_add_dynamic = Button(fig, label = "Add dynamic")
    on(b_add_dynamic.clicks) do _
        filter_prop_name = prop_name[]
        push!(active_filters, (
                Symbol(filter_prop_name),
                row_index[],
                low[],
                hi[],
                limits[filter_prop_name],
                transform_name[]
                )
            )
        reset_selection_slider!()
    end
    top_buttons[1, 1:5] = [genlabel("Filters"), b_clear, b_clear_last, b_add_static, b_add_dynamic]

    # Edge outlines
    if !isnothing(edge_color)
        top_layout[1, N_top] = genlabel("Edges")
        N_top += 1
        edge_toggle = Toggle(top_layout[1, N_top], active = true)
        N_top += 1
    end

    # Transform
    top_layout[1, N_top] = genlabel("Transform")
    N_top += 1
    menu_transform = Menu(top_layout[1, N_top], options = ["none", "abs", "log10", "symlog10", "exp", "10^", "log", ">0", "<0"], prompt = "none")
    on(menu_transform.selection) do s
        transform_name[] = s
        lims[] = get_limits(menu_cscale.selection[], state_index.val, prop_name.val, row_index.val, states, transform_name[])
    end
    N_top += 1

    # Colormap selector at the end
    top_layout[1, N_top] = genlabel("Colormap")
    N_top += 1

    colormaps = [
        "autumn1",
        "balance",
        "batlowK",
        "berlin",
        "brg",
        "commercial",
        "gnuplot",
        "gray1",
        "hawaii",
        "hot",
        "imola",
        "jet",
        "ocean",
        "oslo",
        "rainbow1",
        "romaO",
        "seaborn_icefire_gradient",
        "seaborn_rocket_gradient",
        "tab20",
        "terrain",
        "terrain",
        "turbo",
        "twilight",
        "vanimo",
        "vik",
        "vikO",
        "viridis",
        "winter"
    ]
    cmap_str = "$colormap"
    if !(cmap_str in colormaps)
        push!(colormaps, cmap_str)
    end
    colormaps = sort(colormaps)
    menu_cmap = Menu(top_layout[1, N_top], options = colormaps, default = cmap_str)
    on(menu_cmap.selection) do s
        colormap_name[] = Symbol(s)
    end
    N_top += 1

    # Colormap selector at the end
    top_layout[1, N_top] = genlabel("Scale")
    N_top += 1

    if length(states) > 1
        opt = ["All steps", "All steps, row", "Current step, row", "Current step"]
    else
        opt = ["All rows", "Row"]
    end
    menu_cscale = Menu(
        top_layout[1, N_top],
        options = opt
    )
    function get_limits(s, state_index, pname, row, states, transform_name)
        if s == "All steps"
            new_lims = limits[pname]
        elseif s == "All steps, row"
            new_lims = row_limits[pname][row]
        else
            current_val = states[state_index[]][Symbol(pname)]
            if s == "Current step, row" || s == "Row"
                if current_val isa Vector
                    cstateval = current_val
                else
                    cstateval = view(current_val, row, :)
                end
            else
                @assert s == "Current step" || s == "All rows"
                cstateval = current_val
            end
            new_lims = my_minmax(cstateval, Inf, -Inf)
        end
        lim_lo, lim_hi = transform_plot_limits(new_lims, transform_name)
        if !isfinite(lim_lo)
            lim_lo = -1e30
        end
        if !isfinite(lim_hi)
            lim_hi = 1e30
        end
        if lim_lo ≈ lim_hi
            lim_lo *= 0.999
            lim_hi = lim_hi*1.001 + 1e-16
        end
        return (lim_lo, lim_hi*1.000001)
    end
    on(menu_cscale.selection) do s
        pname = prop_name[]
        row = row_index[]
        lims[] = get_limits(s, state_index, pname, row, states, transform_name[])
    end
    N_top += 1

    N_mid = 1
    top_layout[2, N_mid] = genlabel("View")
    N_mid += 1
    menu_view = Menu(
        top_layout[2, N_mid],
        options = ["Default", "XZ", "YZ", "XY"]
    )
    N_mid += 1
    on(menu_view.selection) do s
        if s == "XZ"
            az = -0.5π
            el = 0.0
        elseif s == "YZ"
            az = 0
            el = 0.0
        elseif s == "XY"
            az = -0.5π
            el = 0.5π
        elseif s == "Default"
            az = 1.275π
            el = 0.125π
        end
        ax.azimuth[] = az
        ax.elevation[] = el
    end


    function loopy()
        start = state_index.val
        if start == nstates
            increment_index(-nstates)
            start = 1
        end
        previndex = start
        for i = start:nstates
            newindex = increment_index()
            if newindex > nstates || previndex != newindex-1
                break
            end
            notify(state_index)
            previndex = newindex
            sleep(1/30)
        end
    end

    fig[5, 1] = buttongrid = GridLayout()
    rewind = Button(fig, label = "⏪")
    on(rewind.clicks) do n
        increment_index(-nstates)
    end
    prev = Button(fig, label = "◀️")
    on(prev.clicks) do n
        increment_index(-1)
    end

    play = Button(fig, label = "⏯️")
    on(play.clicks) do n
        @async loopy()
    end
    next =   Button(fig, label = "▶️")
    on(next.clicks) do n
        increment_index()
    end
    ffwd = Button(fig, label = "⏩")
    on(ffwd.clicks) do n
        increment_index(nstates)
    end
    buttongrid[1, 1:5] = [rewind, prev, play, next, ffwd]

    # Actual plotting call
    if plot_type == :mesh
        # TODO: Not sure if this speeds things up
        tri_c = Makie.to_triangles(primitives.triangulation)
        pts_c = Makie.to_vertices(pts)
        scat = Makie.mesh!(ax, pts_c, tri_c;
            color = ys,
            colorrange = lims,
            highclip = :transparent,
            backlight = 1,
            colormap = cmap,
            transparency = transparency,
            kwarg...
        )
        if !isnothing(edge_color)
            eplt = Jutul.plot_mesh_edges!(ax, grid; color = edge_color, edge_arg...)
            connect!(eplt.visible, edge_toggle.active)
        end
    elseif plot_type == :meshscatter
        sz = 0.8.*primitives.sizes
        npts, d = size(pts)
        if d < 3
            pts = hcat(pts, zeros(npts, 3 - d))
            sz = hcat(sz, ones(npts, 3 - d))
        end 
        sizes = zeros(Makie.Vec3f, size(sz, 1))
        for i in eachindex(sizes)
            sizes[i] = Makie.Vec3f(sz[i, 1], sz[i, 2], sz[i, 3])
        end
        scat = Makie.meshscatter!(ax, pts;
            color = ys,
            colorrange = lims,
            markersize = sizes,
            shading = is_3d,
            colormap = cmap,
            transparency = transparency,
            kwarg...
        )
    elseif plot_type == :lines
        x = pts[:, 1]
        y = pts[:, 2]
        z = pts[:, 3]
        scat = Makie.lines!(ax, x, y, z,
            color = ys,
            linewidth = 15,
            transparency = transparency,
            colormap = cmap,
            colorrange = lims
        )
        txt = primitives.top_text
        if !isnothing(txt)
            top = vec(pts[1, :])
            text!(txt,
                    position = Tuple([top[1], top[2], top[3] + 2.0]),
                    space = :data,
                    align = (:center, :baseline)
                    )
        end
        if primitives.marker_size > 0
            Makie.scatter!(ax, x, y, z, marker_size = primitives.marker_size, color = :black, alpha = 0.5, overdraw = true)
        end
    else
        error("Unsupported plot_type $plot_type")
    end

    Colorbar(fig[3, 1:3], scat, vertical = false)

    if new_window
        Jutul.independent_figure(fig)
    else
        # We might be replacing an existing figure that binds a lot of memory.
        GC.gc()
    end
    return fig
end

function select_data(buffer, current_filter, state, fld, ix, low, high, limits, transform_name, active_filters)
    d = unpack(buffer, state[fld], ix)
    current_active = low > 0.0 || high < 1.0
    @. current_filter = false
    function update_filter!(M::AbstractMatrix, ix, arg...)
        update_filter!(view(M, ix, :), ix, arg...)
    end

    function update_filter!(M::AbstractVector, ix, L, U, low, high)
        for i in eachindex(M)
            val = (M[i] - L)/(U - L)
            cell_hidden = val < low || val > high
            current_filter[i] |= cell_hidden
        end
    end

    for filt in active_filters
        if filt isa Vector{Bool}
            @. current_filter |= filt
        else
            nm, ix, low_dyn, high_dyn, limits_dyn, _ = filt
            L, U = limits_dyn
            filter_d = state[nm]
            update_filter!(filter_d, ix, L, U, low_dyn, high_dyn)
        end
    end

    function apply_filter!(M::AbstractVector)
        for i in eachindex(M)
            if current_filter[i]
                M[i] = NaN
            end
        end
    end
    if transform_name != "none"
        for i in eachindex(d)
            d[i] = plot_transform(d[i], transform_name)
        end
    end

    if current_active
        L, U = limits
        update_filter!(d, 1, L, U, low, high)
    end
    apply_filter!(d)
    return d
end

function unpack(buffer, x, ix)
    ix = min(ix, size(x, 1))
    for j in axes(x, 2)
        buffer[j] = Float64(x[ix, j])
    end
    return buffer
end


function unpack(buffer, x::AbstractVector, ix)
    @. buffer = Float64(x)
    return buffer
end

function generate_colormap(colormap_name, alphamap_name, base_alpha, low, high)
    if colormap_name == :commercial
        cmap = commercial_colormap()
    else
        cmap = to_colormap(colormap_name)
    end
    n = length(cmap)
    if alphamap_name != :no_alpha_map
        if alphamap_name == :linear
            F = x -> x
        elseif alphamap_name == :inv_linear
            F = x -> 1.0 - x
        elseif alphamap_name == :linear_scaled
            F = x -> clamp((x - low)./(high-low), 0.0, 1.0)
        elseif alphamap_name == :inv_linear_scaled
            F = x -> clamp(((1.0 - x) - high)./(low - high), 0.0, 1.0)
        else
            error()
        end
        u = range(0, 1, length = n)
        for (i, c) in enumerate(cmap)
            cmap[i] = Makie.RGBA{Float64}(c.r, c.g, c.b, base_alpha*F(u[i]))
        end
    else
        for (i, c) in enumerate(cmap)
            cmap[i] = Makie.RGBA{Float64}(c.r, c.g, c.b, base_alpha)
        end
    end
    return cmap
end

function basic_3d_figure(resolution = default_jutul_resolution(); z_is_depth = false)
    fig = Figure(size = resolution)
    ax = Axis3(fig[1, 1], zreversed = z_is_depth)
    return (fig, ax)
end

function basic_2d_figure(resolution = default_jutul_resolution(); z_is_depth = false)
    fig = Figure(size = resolution)
    ax = Axis(fig[1, 1])
    return (fig, ax)
end


function symlog10(x)
    # Inspired by matplotlib.scale.SymmetricalLogScale
    # https://matplotlib.org/stable/api/scale_api.html#matplotlib.scale.SymmetricalLogScale
    if x < 1 && x > -1
        return x
    else
        return sign(x)*(log10(abs(x))+1)
    end
end

function plot_transform(x, name)
    if name != "none"
        if name == "abs"
            x = abs(x)
        elseif name == "log10"
            x = x > 0 ? log10(x) : NaN
        elseif name == "log"
            x = x > 0 ? log(x) : NaN
        elseif name == "symlog10"
            x = symlog10(x)
        elseif name == "exp"
            x = exp(x)
        elseif name == "10^"
            x = 10^x
        elseif name == ">0"
            x = x > 0 ? x : NaN
        elseif name == "<0"
            x = x < 0 ? x : NaN
        else
            error("Unknown transform $name")
        end
    end
    if !isfinite(x)
        x = NaN
    end
    return x
end


function transform_plot_limits(lims, name)
    low, hi = lims
    if name != "none"
        if name == "abs"
            low = 0.0
            hi = abs(hi)
        elseif name == "log10" || name == log
            if low < 0.0
                low = -1e6
            else
                low = plot_transform(low, name)
            end
            hi = plot_transform(hi, name)
        elseif name == ">0"
            low = 0.0
        elseif name == "<0"
            hi = 0.0
        else
            low = plot_transform(low, name)
            hi = plot_transform(hi, name)
        end
    end
    if hi <= low
        hi = low + 1e-12
    end
    return (low, hi)
end

function Jutul.plot_multimodel_interactive_impl(model, states, model_keys = keys(model.models); plot_type = :mesh, shift = Dict(), kwarg...)
    n = length(model_keys)
    primitives = Vector{Any}(undef, n)
    ncells = zeros(Int64, n)
    active = BitArray(undef, n)
    active .= false
    for (i, k) in enumerate(model_keys)
        p = physical_representation(model[k].data_domain)
        if isnothing(p)
            keep = false
        else
            primitive = plot_primitives(p, plot_type)
            keep = !isnothing(primitive)
            if keep
                nc = maximum(primitive.mapper.indices.Cells)
                ncells[i] = nc
                primitives[i] = primitive
                keep = keep && nc > 0
            end
        end
        active[i] = keep
    end
    model_keys = model_keys[active]
    primitives = primitives[active]
    ncells = ncells[active]
    # Remap states so that we have NaN padded versions
    offsets = cumsum(vcat([1], ncells))
    states_mapped = Vector{Dict{Symbol, Any}}()
    # Find all possible state fields
    all_state_fields = []
    state = states[1]
    for (i, model_key) in enumerate(model_keys)
        nc = ncells[i]
        for (k, v) in state[model_key]
            valid_vector = v isa AbstractVector && length(v) == nc
            valid_matrix = v isa AbstractMatrix && size(v, 2) == nc

            if valid_vector 
                push!(all_state_fields, (k, 1))
            elseif valid_matrix
                push!(all_state_fields, (k, size(v, 1)))
            end
        end
    end
    # Create flattened states with NaN for missing data
    total_number_of_cells = sum(ncells)
    new_states = Vector{Dict{Symbol, Any}}()
    for state in states
        new_state = Dict{Symbol, Any}()
        for (state_field, d) in all_state_fields
            data = zeros(d, total_number_of_cells)
            data .= NaN
            for (i, model_key) in enumerate(model_keys)
                state_m = state[model_key]
                if haskey(state_m, state_field)
                    old_data = state_m[state_field]
                    data[:, offsets[i]:(offsets[i+1]-1)] = old_data
                end
            end
            new_state[state_field] = data
        end
        push!(new_states, new_state)
    end
    # Merge triangulations
    face_index = Vector{Int64}()
    points = map(x -> x.points, primitives)
    tri = map(x -> x.triangulation, primitives)
    cell_index = map(x -> x.mapper.indices.Cells, primitives)
    offset = 0
    for (pts, T, k) in zip(points, tri, model_keys)
        @. T += offset
        if haskey(shift, k)
            for (i, dx) in enumerate(shift[k])
                @. pts[:, i] += dx
            end
        end
        offset += size(pts, 1)
    end
    offset = 0
    for (nc, cix) in zip(ncells, cell_index)
        @. cix += offset
        offset += nc
    end
    tri = vcat(tri...)
    points = vcat(points...)
    cell_index = vcat(cell_index...)

    mapper = (
                Cells = (cell_data) -> cell_data[cell_index],
                Faces = (face_data) -> face_data[face_index],
                indices = (Cells = cell_index, Faces = face_index)
              )
    acc_primitives = (points = points, triangulation = tri, mapper = mapper)
    plot_interactive(total_number_of_cells, new_states, primitives = acc_primitives; kwarg...)
end

function Jutul.plotting_check_interactive(; warn = true)
    backend_name = "$(Makie.current_backend())"
    if backend_name != "GLMakie"
        if warn
            msg = "Currently active Makie backend $backend_name may not be interactive or fully supported.\nGLMakie is recommended for Jutul's interactive plots. To install:\n\tusing Pkg; Pkg.add(\"GLMakie\")\nTo use:\n\tusing GLMakie\n\tGLMakie.activate!()\nYou can then retry your plotting command."
            @warn msg
        end
        return false
    end
    return true
end

function commercial_colormap()
    blue =   (0, 0, 1)
    cyan =   (0, 1, 1)
    green =  (0, 1, 0)
    yellow = (1, 1, 0)
    red =    (1, 0, 0)

    function simple_interp(F_0, F_1, x)
        v = F_0 .+ (F_1 .- F_0).*x
        return Makie.RGB(v...)
    end
    cmap = Vector{typeof(Makie.RGB(0, 0, 0))}()
    nsteps = [30, 25, 25, 20]
    colors = (blue, cyan, green, yellow, red)
    for (i, nstep) in enumerate(nsteps)
        c1 = colors[i]
        c2 = colors[i+1]
        for dx in range(0.0, 1.0, nstep)
            push!(cmap, simple_interp(c1, c2, dx))
        end
    end
    return cmap
end

function my_minmax(di, mv, Mv)
    for v in di
        if !isfinite(v)
            continue
        end
        mv = min(mv, v)
        Mv = max(Mv, v)
    end
    return (mv, Mv)
end
