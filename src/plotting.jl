function plot_interactive(grid, states; plot_type = nothing)
    pts, tri, mapper = triangulate_outer_surface(grid)

    fig = Figure()
    data = states[1]
    datakeys = collect(keys(data))
    state_index = Node{Int64}(1)
    prop_name = Node{Symbol}(datakeys[1])
    loop_mode = Node{Int64}(0)

    menu = Menu(fig, options = datakeys)
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
    fig[1, 1] = vgrid!(
        Label(fig, "Property", width = nothing),
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
        ax = Axis3(fig[1, 2])
    else
        ax = Axis(fig[1, 2])
    end
    ys = @lift(mapper.Cells(select_data(states[$state_index], $prop_name)))
    scat = Makie.mesh!(ax, pts, tri, color = ys)
    # cb = Colorbar(fig[1, 3], scat, vertical = true, width = 30)

    on(menu.selection) do s
        prop_name[] = s
        autolimits!(ax)
    end
    # on(menu2.selection) do s
    # end
    # menu2.is_open = true

    function loop(a)
        # looping = !looping
        # println("Loop function called")
        if false
            if loop_mode.val > 0
                # println("Doing loop")
                start = state_index.val
                if start == nstates
                    start = 1
                end
                for i = start:nstates
                    newindex = increment_index()
                    if newindex > nstates
                        break
                    end
                    notify(state_index)
                    force_update!()
                    sleep(1/30)
                end
            end
        end
    end

    # @lift(loop($loop_mode))

    fig[2, 1] = buttongrid = GridLayout(tellwidth = false)
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
        println("Play button is not implemented.")
        # loop_mode[] = loop_mode.val + 1
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
    return fig
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
        return vec(d)
    else
        return get_vector(d[1, :])
    end
end
