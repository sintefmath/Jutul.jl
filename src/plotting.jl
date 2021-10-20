export plot_well!
function plot_interactive(grid, states; plot_type = nothing, wells = nothing, kwarg...)
    pts, tri, mapper = triangulate_outer_surface(grid)

    fig = Figure()
    data = states[1]
    datakeys = collect(keys(data))
    initial_prop = datakeys[1]
    state_index = Node{Int64}(1)
    prop_name = Node{Symbol}(initial_prop)
    loop_mode = Node{Int64}(0)

    menu = Menu(fig, options = datakeys, prompt = String(initial_prop))
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

    # funcs = [sqrt, x->x^2, sin, cos]
    # menu2 = Menu(fig, options = zip(["Square Root", "Square", "Sine", "Cosine"], funcs))
    fig[2, 3] = vgrid!(
        #Label(fig, "Property", width = nothing),
        menu,
        # Label(fig, "Function", width = nothing),
        # menu2
        ; tellheight = false, width = 200)
    
    sl_x = Slider(fig[2, 2], range = 1:nstates, value = state_index, snap = true)
    # point = sl_x.value
    on(sl_x.selected_index) do n
        state_index[] = sl_x.selected_index.val
    end
    if size(pts, 2) == 3
        ax = Axis3(fig[1, 1:2])
    else
        ax = Axis(fig[1, 1:2])
    end
    is_3d = size(pts, 2) == 3
    ys = @lift(mapper.Cells(select_data(states[$state_index], $prop_name)))
    scat = Makie.mesh!(ax, pts, tri, color = ys, size = 60; shading = is_3d, kwarg...)
    cb = Colorbar(fig[1, 3], scat)

    on(menu.selection) do s
        prop_name[] = s
        autolimits!(ax)
    end

    function loopy()
        start = state_index.val
        if start == nstates
            start = 1
        end
        previndex = start
        for i = start:nstates
            newindex = increment_index()
            if newindex > nstates || previndex != newindex-1
                break
            end
            notify(state_index)
            force_update!()
            previndex = newindex
            sleep(1/30)
        end
    end

    fig[2, 1] = buttongrid = GridLayout()
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
    buttons = buttongrid[1, 1:5] = [rewind, prev, play, next, ffwd]
    
    display(fig)
    return fig, ax
end

function select_data(state, fld)
    v = state[fld]
    return get_vector(v)
end

function get_vector(d::Vector)
    return d
end

function get_vector(d::Matrix)
    if size(d, 1) == 1 || size(d, 2) == 1
        return get_vector(vec(d))
    else
        return get_vector(d[1, :])
    end
end

function plot_well!(ax, g, w; color = :darkred, textcolor = nothing, linewidth = 5, top_factor = 0.2, textscale = 2.5e-2, kwarg...)
    if isnothing(textcolor)
        textcolor = color
    end
    raw = g.data
    coord_range(i) = maximum(raw.cells.centroids[:, i]) - minimum(raw.cells.centroids[:, i])

    z = raw.cells.centroids[:, 3]
    bottom = maximum(z)
    top = minimum(z)

    xrng = coord_range(1)
    yrng = coord_range(2)
    textsize = textscale*(xrng + yrng)/2

    rng = top - bottom
    s = top + top_factor*rng

    c = vec(Int64.(w["cells"]))
    pts = raw.cells.centroids[[c[1], c...], :]
    pts[1, 3] = s

    l = pts[1, :]
    text!(w["name"], position = Tuple([l[1], l[2], -l[3]]), space = :data, color = textcolor, align = (:center, :baseline), textsize = textsize)
    lines!(ax, vec(pts[:, 1]), vec(pts[:, 2]), -vec(pts[:, 3]), linewidth = linewidth, color = color, kwarg...)
end
