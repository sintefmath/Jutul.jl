import Jutul: plot_interactive_impl


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

function plot_interactive_impl(grid, states; plot_type = nothing,
                                        primitives = nothing,
                                        transparency = false,
                                        resolution = default_jutul_resolution(),
                                        alpha = 1.0,
                                        title = "",
                                        transform = "none",
                                        colormap = :viridis,
                                        alphamap = :no_alpha_map,
                                        kwarg...)
    has_primitives = !isnothing(primitives)
    active_filters = []
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

    fig = Figure(resolution = resolution)
    if states isa AbstractDict
        states = [states]
    end
    if eltype(states)<:Number && (length(states) == nc || size(states, 1) == nc)
        states = [Dict(:Data => states)]
    end
    data = states[1]
    labels = Vector{String}()
    limits = Dict()
    for k in keys(data)
        d = data[k]
        is_valid_vec = d isa AbstractVector && length(d) == nc
        is_valid_mat = d isa AbstractMatrix && size(d, 2) == nc
        if eltype(d)<:Real && (is_valid_vec || is_valid_mat) 
            push!(labels, "$k")
            mv = Inf
            Mv = -Inf
            for s in states
                di = s[k]
                mv = min(minimum(x -> isnan(x) ? Inf : x, di), mv)
                Mv = max(maximum(x -> isnan(x) ? -Inf : x, di), Mv)
            end
            if mv == Mv
                Mv = 1.01*mv + 1e-12
            end
            limits["$k"] = (mv, Mv)
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
    initial_prop = datakeys[1]
    state_index = Observable{Int64}(1)
    row_index = Observable{Int64}(1)
    prop_name = Observable{Any}(initial_prop)
    transform_name = Observable{String}(transform)
    lims = Observable(limits[initial_prop])
    menu = Menu(fig, options = datakeys, prompt = initial_prop)
    menu_2 = Menu(fig, options = get_valid_rows("$initial_prop"), prompt = "1", width = 60)

    nstates = length(states)

    function change_index(ix)
        tmp = max(min(ix, nstates), 1)
        sl_x.selected_index = tmp
        state_index[] = tmp
        notify(state_index)
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
        low[] = x[1]
        hi[] = x[2]
    end
    # point = sl_x.value
    on(sl_x.selected_index) do n
        val = sl_x.selected_index.val
        state_index[] = val
    end
    is_3d = size(pts, 2) == 3
    if is_3d
        make_axis = Axis3
    else
        make_axis = Axis
    end
    ax = make_axis(fig[2, 1:3], title = title)

    # Selection of data
    ys = @lift(
                mapper.Cells(
                    select_data(
                        current_filter,
                        states[$state_index],
                        Symbol($prop_name),
                        $row_index,
                        $low,
                        $hi,
                        limits[$prop_name],
                        $transform_name,
                        active_filters
                        )
                )
            )
    # Selection of colormap
    colormap_name = Observable(colormap)
    alphamap_name = Observable(alphamap)
    cmap = @lift(generate_colormap($colormap_name, $alphamap_name, alpha, $low, $hi))

    # Menu for field to plot
    on(menu.selection) do s
        rows = get_valid_rows(s)
        msel =  menu_2.selection[]
        if isnothing(msel)
            old = 1
        else
            old = parse(Int64, msel)
        end
        nextn = min(old, length(rows))
        prop_name[] = s
        row_index[] = nextn
        notify(prop_name)
        notify(menu_2.selection)
        menu_2.options = rows
        menu_2.selection[] = "$nextn"
        lims[] = transform_plot_limits(limits[s], transform_name[])
    end
    # Row of dataset selector
    on(menu_2.selection) do s
        if isnothing(s)
            s = "1"
        end
        row_index[] = parse(Int64, s)
    end
    # Top row
    fig[1, :] = top_layout = GridLayout(tellwidth = false)
    N_top = 0

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

    # Transform
    top_layout[1, N_top] = genlabel("Transform")
    N_top += 1
    menu_transform = Menu(top_layout[1, N_top], options = ["none", "abs", "log10", "symlog10", "exp", "10^", "log", "≥0", "≤0"], prompt = "none")
    on(menu_transform.selection) do s
        transform_name[] = s
        old_lims = limits[prop_name[]]
        new_lims = transform_plot_limits(old_lims, transform_name[])
        lims[] = new_lims
    end
    N_top += 1

    # Colormap selector at the end
    top_layout[1, N_top] = genlabel("Colormap")
    N_top += 1

    colormaps = [
        "viridis",
        "turbo",
        "oslo",
        "jet",
        "balance",
        "autumn1",
        "hot",
        "winter",
        "terrain",
        "gnuplot",
        "ocean",
        "vik",
        "twilight",
        "terrain",
        "berlin",
        "hawaii",
        "seaborn_icefire_gradient",
        "seaborn_rocket_gradient",
        "imola",
        "gray1",
        "rainbow1",
        "tab20"
        ]
    cmap_str = "$colormap"
    if !(cmap_str in colormaps)
        push!(colormaps, cmap_str)
    end
    menu_cmap = Menu(top_layout[1, N_top], options = colormaps, prompt = cmap_str)
    on(menu_cmap.selection) do s
        colormap_name[] = Symbol(s)
    end
    N_top += 1

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
        tri = primitives.triangulation
        scat = Makie.mesh!(ax, pts, tri; color = ys,
                                        colorrange = lims,
                                        size = 60,
                                        shading = is_3d,
                                        colormap = cmap,
                                        transparency = transparency,
                                        kwarg...)
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
        scat = Makie.meshscatter!(ax, pts; color = ys,
                                        colorrange = lims,
                                        markersize = sizes,
                                        shading = is_3d,
                                        colormap = cmap,
                                        transparency = transparency,
                                        kwarg...)
    elseif plot_type == :lines
        x = pts[:, 1]
        y = pts[:, 2]
        z = pts[:, 3]
        scat = Makie.lines!(ax, x, y, z, color = ys,
                                                    linewidth = 15,
                                                    transparency = transparency,
                                                    colormap = cmap,
                                                    colorrange = lims)
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

    # display(GLMakie.Screen(), fig)
    return fig#, ax
end

function select_data(current_filter, state, fld, ix, low, high, limits, transform_name, active_filters)
    d = unpack(state[fld], ix)
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

    if current_active
        L, U = limits
        update_filter!(d, 1, L, U, low, high)
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
    apply_filter!(d)
    return d
end

unpack(x, ix) = x[min(ix, size(x, 1)), :]
unpack(x::AbstractVector, ix) = copy(x)

function generate_colormap(colormap_name, alphamap_name, base_alpha, low, high)
    cmap = to_colormap(colormap_name)
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
    end
    return cmap
end

function basic_3d_figure()
    fig = Figure()
    ax = Axis3(fig[1, 1])
    return (fig, ax)
end

