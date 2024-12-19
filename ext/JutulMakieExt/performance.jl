function Jutul.plot_solve_breakdown(allreports, names; per_it = false, include_local_solves = nothing, t_scale = ("s", 1.0))
    t_unit, t_num = t_scale
    if per_it
        plot_title = "Time per iteration"
        to_plot = x -> [x.assembly/x.its, x.subdomains/x.its, x.solve/x.its, x.total/x.its]./t_num
    else
        plot_title = "Total time"
        to_plot = x -> [x.assembly, x.subdomains, x.solve, x.total]./t_num
    end
    labels = ["Assembly", "Local solves", "Linear solve", "Total"]

    D = map(x -> to_plot(timing_breakdown(x)), allreports)
    ndata = length(D)
    nel = length(D[1])

    if isnothing(include_local_solves)
        include_local_solves = sum(x -> x[2], D) > 0.0
    end
    colors = Makie.wong_colors(1.0)
    if include_local_solves
        colors = colors[[1, 2, 3, 6]]
    else
        subs = [1, 3, 4]
        colors = colors[[1, 2, 3]]
        labels = labels[subs]
        nel = length(subs)
        D = map(x -> x[subs], D)
    end

    h = vcat(D...)
    x = ones(nel)
    for i = 2:ndata
        x = vcat(x, i*ones(nel))
    end
    grp = repeat(1:nel, ndata)

    fig = Figure()
    ax = Axis(fig[1,1], xticks = (1:ndata, names), ylabel = "Time [$t_unit]", title = plot_title)
    barplot!(ax, x, h,
            dodge = grp,
            color = colors[grp])

    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    title = nothing# "Legend"
    Legend(fig[2,1], elements, labels, title, orientation = :horizontal)
    display(fig)
    return (fig, D)
end

function Jutul.plot_cumulative_solve(allreports, arg...; kwarg...)
    fig = Figure()
    ax, alldata, t = Jutul.plot_cumulative_solve!(fig[1, 1], allreports, arg...; kwarg...)
    display(fig)
    return (fig, ax, alldata, t)
end

function Jutul.plot_cumulative_solve!(f, allreports, dt = nothing, names = nothing; 
        use_time = false,
        use_title = true,
        linewidth = 3.5,
        cumulative = true,
        linestyles = missing,
        colormap = :tab20,
        t_scale = ("s", 1.0),
        axis_arg = NamedTuple(),
        x_is_time = true,
        legend = true,
        ministeps = false,
        title = nothing,
        scatter_points = true,
        kwarg...
    )
    function get_linestyle(i)
        if ismissing(linestyles)
            return nothing
        else
            return linestyles[i]
        end
    end
    if first(allreports) isa AbstractDict && haskey(first(allreports), :ministeps)
        allreports = [allreports]
    end
    if ministeps
        allreports = map(Jutul.reports_to_ministep_reports, allreports)
    end
    if isnothing(dt)
        dt = map(report_timesteps, allreports)
    end
    r_rep = map(x -> timing_breakdown(x, reduce = false), allreports)
    if use_time
        t_unit, t_num = t_scale
        F = D -> map(x -> x.total/t_num, D)
        yl = "Wall time [$t_unit]"
        tit = "Runtime"
    else
        F = D -> map(x -> x.its, D)
        yl = "Iterations"
        tit = "Nonlinear iterations"
    end
    if !isnothing(title)
        tit = "$title: $tit"
    end
    if !use_title
        tit = ""
    end
    if x_is_time
        xl = "Time [years]"
    else
        xl = "Step"
    end

    ax = Axis(f; xlabel = xl, title = tit, ylabel = yl, axis_arg...)
    if cumulative
        get_data = x -> cumsum(vcat(0, F(x)))
        t = map(dt -> cumsum(vcat(0, dt))/(3600*24*365), dt)
    else
        get_data = x -> F(x)
        t = map(dt -> cumsum(dt)/(3600*24*365), dt)
    end
    if !x_is_time
        t = map(t -> eachindex(t), dt)
    end
    names_missing = isnothing(names)
    if names_missing
        names = map(x -> "dataset $x", eachindex(r_rep))
    end
    alldata = []
    colors = to_colormap(colormap)
    n_rep = length(r_rep)
    for i in eachindex(r_rep)
        c = colors[mod(i-1, n_rep)+1]
        data_i = get_data(r_rep[i])
        push!(alldata, data_i)
        lstyle = get_linestyle(i)
        skip_line = ismissing(lstyle)
        if !skip_line
            lines!(ax, t[i], data_i,
                label = names[i],
                linewidth = linewidth,
                color = c,
                linestyle = lstyle;
                kwarg...)
        end
        if scatter_points
            if skip_line
                scatter!(ax, t[i], data_i, color = c, label = names[i])
            else
                scatter!(ax, t[i], data_i, color = c)
            end
        end
    end
    if length(r_rep) > 1 && !names_missing && legend
        axislegend(ax, position = :lt)
    end
    return (ax, alldata, t)
end


function Jutul.plot_linear_convergence(report; kwarg...)
    fig = Figure()
    ax = Axis(fig[1, 1], yscale = log10; ylabel = "Residual", xlabel = "Linear iterations", kwarg...)
    plot_linear_convergence!(ax, report)
    return fig
end

function Jutul.plot_linear_convergence!(ax, report::AbstractDict)
    if haskey(report, :ministeps)
        plot_linear_convergence!(ax, report[:ministeps])
    elseif haskey(report, :steps)
        plot_linear_convergence!(ax, report[:steps])
    elseif haskey(report, :linear_solver)
        r = report[:linear_solver].residuals
        lines!(ax, 1:length(r), r, alpha = 0.5)
    end
end

function Jutul.plot_linear_convergence!(ax, reports::Vector)
    for r in reports
        plot_linear_convergence!(ax, r)
    end
end
