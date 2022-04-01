export plot_well!, plot_interactive, plot_well_results
using .GLMakie
using ColorSchemes
function plot_interactive(grid, states; plot_type = nothing, wells = nothing, kwarg...)
    pts, tri, mapper = triangulate_outer_surface(grid)

    fig = Figure()
    data = states[1]
    labels = Vector{String}()
    pos = Vector{Tuple{Symbol, Integer}}()
    limits = Dict()
    for k in keys(data)
        d = data[k]
        if isa(d, AbstractVector)
            push!(labels, "$k")
            push!(pos, (k, 1))
        else
            for i = 1:size(d, 1)
                push!(labels, "$k: $i")
                push!(pos, (k, i))
            end
        end
        mv = Inf
        Mv = -Inf
        for s in states
            di = s[k]
            mv = min(minimum(di), mv)
            Mv = max(maximum(di), Mv)
        end
        limits[k] = (mv, Mv)
    end
    datakeys = collect(zip(labels, pos))
    initial_prop = datakeys[1]
    state_index = Observable{Int64}(1)
    prop_name = Observable{Any}(initial_prop[2])
    lims = Observable(limits[get_label(initial_prop[2])])
    menu = Menu(fig, options = datakeys, prompt = initial_prop[1])
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

    fig[3, 3] = vgrid!(
        #Label(fig, "Property", width = nothing),
        menu,
        # Label(fig, "Function", width = nothing),
        # menu2
        ; tellheight = false, width = 300)
    
    sl_x = Slider(fig[3, 2], range = 1:nstates, value = state_index, snap = true)
    # point = sl_x.value
    on(sl_x.selected_index) do n
        val = sl_x.selected_index.val
        state_index[] = val
    end
    if size(pts, 2) == 3
        ax = Axis3(fig[1, 1:3])
    else
        ax = Axis(fig[1, 1:3])
    end
    is_3d = size(pts, 2) == 3
    ys = @lift(mapper.Cells(select_data(states[$state_index], $prop_name)))
    scat = Makie.mesh!(ax, pts, tri, color = ys, colorrange = lims, size = 60; shading = is_3d, kwarg...)
    cb = Colorbar(fig[2, 1:3], scat, vertical = false)

    on(menu.selection) do s
        prop_name[] = s
        pos = get_label(s)
        lims[] = limits[pos]
        # autolimits!(ax)
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

    fig[3, 1] = buttongrid = GridLayout()
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

get_label(x::Tuple) = x[1]
get_label(x) = x

select_data(state, fld::Tuple) = unpack(state[get_label(fld)], fld[2])

unpack(x, ix) = x[ix, :]
unpack(x::AbstractVector, ix) = x


function plot_well!(ax, g, w; color = :darkred, textcolor = nothing, linewidth = 5, top_factor = 0.2, textscale = 2.5e-2, kwarg...)
    if isnothing(textcolor)
        textcolor = color
    end
    raw = g.data
    coord_range(i) = maximum(raw.cells.centroids[:, i]) - minimum(raw.cells.centroids[:, i])

    if size(raw.cells.centroids, 2) == 3
        z = raw.cells.centroids[:, 3]
    else
        z = [0.0, 1.0]
    end
    bottom = maximum(z)
    top = minimum(z)

    xrng = coord_range(1)
    yrng = coord_range(2)
    textsize = textscale*(xrng + yrng)/2

    rng = top - bottom
    s = top + top_factor*rng

    wc = w["cells"]
    if !isa(wc, AbstractArray)
        wc = [wc]
    end
    c = vec(Int64.(wc))
    pts = raw.cells.centroids[[c[1], c...], :]
    if size(pts, 2) == 2
        pts = hcat(pts, zeros(size(pts, 1)))
    end
    pts[1, 3] = s

    l = pts[1, :]
    text!(w["name"], position = Tuple([l[1], l[2], -l[3]]), space = :data, color = textcolor, align = (:center, :baseline), textsize = textsize)
    lines!(ax, vec(pts[:, 1]), vec(pts[:, 2]), -vec(pts[:, 3]), linewidth = linewidth, color = color, kwarg...)
end

export plot_well_results
function plot_well_results(well_data::Dict, arg...; name = "Data", kwarg...)
    plot_well_results([well_data], arg...; names = [name], kwarg...)
end

function plot_well_results(well_data::Vector, time = nothing; start_date::Union{Nothing, DateTime} = nothing,
                                                              names =["Dataset $i" for i in 1:length(well_data)], 
                                                              linewidth = 3, cmap = nothing, 
                                                              styles = [nothing, :dash, :scatter, :dashdot, :dot, :dashdotdot],
                                                              kwarg...)
    # Figure part
    ndata = length(well_data)
    @assert ndata <= length(styles) "Can't plot more datasets than styles provided"
    fig = Figure()
    if isnothing(time)
        t_l = "Time-step"
        @assert isnothing(start_date) "start_date does not make sense in the absence of time-steps"
    elseif isnothing(start_date)
        t_l = "Time [days]"
    else
        t_l = "Date"
    end
    ax = Axis(fig[1, 1], xlabel = t_l)

    wd = first(well_data)
    # Selected well
    wells = sort!(collect(keys(wd)))
    nw = length(wells)
    if isnothing(cmap)
        cmap = cgrad(:Paired_12, nw, categorical=true)
    end
    wellstr = [String(x) for x in wells]

    # Type of plot (bhp, rate...)
    responses = collect(keys(wd[first(wells)]))
    respstr = [String(x) for x in responses]
    response_ix = Observable(1)
    type_menu = Menu(fig, options = respstr, prompt = respstr[1])

    on(type_menu.selection) do s
        val = findfirst(isequal(s), respstr)
        response_ix[] = val
        autolimits!(ax)
    end
    fig[2, 2:3] = hgrid!(
        type_menu)

    b_xlim = Button(fig, label = "Reset x")
    on(b_xlim.clicks) do n
        reset_limits!(ax; xauto = true, yauto = false)
    end
    b_ylim = Button(fig, label = "Reset y")
    on(b_ylim.clicks) do n
        reset_limits!(ax; xauto = false, yauto = true)
    end
    buttongrid = GridLayout(tellwidth = false)
    buttongrid[1, 1] = b_xlim
    buttongrid[1, 2] = b_ylim



    # Lay out and do plotting
    fig[2, 1] = buttongrid
    function get_data(wix, rix, dataix)
        tmp = well_data[dataix][wells[wix]][responses[rix]]
        return tmp
    end

    sample = map(x -> get_data(1, 1, x), 1:ndata)
    nsample = map(length, sample)
    if isnothing(time)
        time = map(s -> 1:length(s), sample)
    else
        if eltype(time)<:AbstractFloat || eltype(time)<:Date
            time = repeat([time], ndata)
        end
    end

    newtime = []
    for i = 1:ndata
        T = time[i]
        nt = length(T)
        ns = nsample[i]
        @assert nt == ns "Series $i: Recieved $nt steps, but wells had $ns results."
        if eltype(T)<:AbstractFloat
            # Scale to days
            if isnothing(start_date)
                @. T /= (3600*24)
            else
                T = @. Microsecond(ceil(T*1e6)) + start_date
            end
            push!(newtime, T)
        end
    end
    time = newtime

    lighten(x) = GLMakie.ARGB(x.r, x.g, x.b, 0.2)
    toggles = [Toggle(fig, active = true, buttoncolor = cmap[i], framecolor_active = lighten(cmap[i])) for i in eachindex(wells)]
    labels = [Label(fig, w) for w in wellstr]

    tmp = hcat(toggles, labels)
    bgrid = tmp
    N = size(bgrid, 1)
    M = div(N, 2, RoundUp)
    fig[1, 2] = grid!(bgrid[1:M, :], tellheight = false)
    fig[1, 3] = grid!(bgrid[(M+1):N, :], tellheight = false)

    lineh = []
    for dix = 1:ndata
        T = time[dix]
        for i in 1:nw
            d = @lift(get_data(i, $response_ix, dix))
            style = styles[dix]
            if style == :scatter
                h = scatter!(ax, T, d, color = cmap[i], linewidth = linewidth, marker = :circle)
            else
                h = lines!(ax, T, d, linewidth = linewidth, linestyle = style, color = cmap[i])
            end
            t = toggles[i]
            connect!(h.visible, t.active)
            push!(lineh, h)
        end
    end
    if ndata > 1
        elems = []
        for i = 1:ndata
            style = styles[i]
            if style == :scatter
                el = MarkerElement(color = :black, linewidth = linewidth, marker = :circle)
            else
                el = LineElement(color = :black, linestyle = styles[i], linewidth = linewidth)
            end
            push!(elems, el)
        end
        fig[1, 1] = Legend(fig, elems, names,
                        tellheight = false,
                        tellwidth = false,
                        margin = (10, 10, 10, 10),
                        halign = :left, valign = :top, orientation = :horizontal
        )
    end

    return fig
end
