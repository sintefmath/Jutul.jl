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
    ax, plt, alldata, t = Jutul.plot_cumulative_solve!(fig[1, 1], allreports, arg...; kwarg...)
    display(fig)
    return (fig, ax, alldata, t)
end

function Jutul.plot_cumulative_solve!(f, allreports, dt = nothing, names = nothing; 
        use_time = false,
        use_title = true,
        linewidth = 3.5,
        linestyles = missing,
        t_scale = ("s", 1.0),
        axis_arg = NamedTuple(),
        legend = true,
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
    if isnothing(dt)
        dt = report_timesteps(first(allreports))
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
    t = cumsum(vcat(0, dt))/(3600*24*365)

    ax = Axis(f; xlabel = "Time [years]", title = tit, ylabel = yl, axis_arg...)
    get_data = x -> cumsum(vcat(0, F(x)))
    names_missing = isnothing(names)
    if names_missing
        names = map(x -> "dataset $x", eachindex(r_rep))
    end
    alldata = []
    for i in eachindex(r_rep)
        data_i = get_data(r_rep[i])
        push!(alldata, data_i)
        lines!(ax, t, data_i, label = names[i], linewidth = linewidth, linestyle = get_linestyle(i), kwarg...)
        if scatter_points
            scatter!(ax, t, data_i)
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
