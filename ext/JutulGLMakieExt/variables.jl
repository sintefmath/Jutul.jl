function Jutul.plot_secondary_variables(model::SimulationModel; kwarg...)
    Jutul.plot_secondary_variables(MultiModel((model = model, )); kwarg...)
end

function Jutul.plot_secondary_variables(model::MultiModel; linewidth = 4, kwarg...)
    data = Dict{String, Any}()
    nregmax = 1
    count = 0
    for (k, m) in pairs(model.models)
        for (vname, var) in Jutul.get_secondary_variables(m)
            d = line_plot_data(m, var)
            if !isnothing(d)
                if d isa JutulLinePlotData
                    d = [d]
                end
                data["$k.$vname"] = d
                if d isa Matrix
                    nregmax = max(nregmax, size(d, 2))
                end
                count += 1
            end
        end
    end
    if count == 0
        @info "No plottable variables found in model."
        return
    end
    fig = Figure()
    grid = GridLayout(fig[1, 1])
    ts = 22.0
    # Property selection
    dkeys = keys(data)
    default = first(dkeys)
    Label(grid[1, 1], "Select property", fontsize = ts)
    m = Menu(grid[2, 1], options = dkeys, default = default, width = 500, fontsize = ts)
    # Region selection
    Label(grid[3, 1], "Select region", fontsize = ts)
    m2 = Menu(grid[4, 1], options = ["All", "Each", ["$i" for i in 1:nregmax]...], default = "All", width = 300, fontsize = ts)
    # Line width
    s = Slider(grid[6, 1], range = range(0.1, 10, step = 0.1), startvalue = linewidth)
    labeltext = lift(s.value) do int
        "Linewidth = $int"
    end
    Label(grid[5, 1], labeltext, fontsize = ts)
    # Actually do stuff
    # Label(grid[8, 1], "Generate plot")
    b = Button(grid[9, 1], label = "Plot variable", buttoncolor = RGBf(0.5, 0.94, 0.5), fontsize = 30.0, buttoncolor_hover = RGBf(0.1, 0.94, 0.1), buttoncolor_active = RGBf(0.1, 0.94, 0.1))

    on(b.clicks) do ix
        d = data[m.selection[]]
        reg = m2.selection[]
        @async begin
            sleep(0.1)
            function plot_by_reg(regions)
                plot_jutul_line_data!(d; regions = regions, linewidth = s.value[])
            end
            if reg == "All"
                plot_by_reg(axes(d, 2))
            elseif reg == "Each"
                for reg in axes(d, 2)
                    plot_by_reg(reg)
                end
            else
                plot_by_reg(min(parse(Int64, reg), size(d, 2)))
            end
        end
    end
    display(fig)
    return fig
end

function plot_jutul_line_data!(data::JutulLinePlotData; kwarg...)
    plot_jutul_line_data!([data]; kwarg...)
end

function plot_jutul_line_data!(data; resolution = default_jutul_resolution(), linewidth = 4, regions = axes(data, 2), kwarg...)
    fig = Figure(resolution = resolution)
    colors = Makie.wong_colors()
    for (col, regix) in enumerate(regions)
        for i in axes(data, 1)
            d = data[i, regix]
            ax = Axis(fig[i, col], xlabel = d.xlabel, ylabel = d.ylabel, title = "$(d.title) region $regix")
            ix = 1
            for (x, y, lbl) in zip(d.xdata, d.ydata, d.datalabels)
                c = colors[mod(ix, 7) + 1]
                lines!(ax, x, y; color = c, linewidth = linewidth, label = lbl, kwarg...)
                ix += 1
            end
            axislegend()
        end
    end
    display(GLMakie.Screen(), fig)
end
