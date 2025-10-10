"""
    plot_solve_breakdown(allreports, names; per_it = false, include_local_solves = nothing, t_scale = ("s", 1.0))

Plot a breakdown of solver performance timing for multiple simulation reports.

# Arguments
- `allreports`: Vector of simulation reports to analyze
- `names`: Vector of names for each report (used for labeling)

# Keyword Arguments
- `per_it = false`: If `true`, show time per iteration instead of total time
- `include_local_solves = nothing`: Whether to include local solve timing (auto-detected if `nothing`)
- `t_scale = ("s", 1.0)`: Time scale unit and conversion factor for display

# Returns
- `(fig, D)`: Figure object and processed timing data

This function creates a bar chart showing the breakdown of computational time spent
in different parts of the simulation: assembly, local solves, linear solve, and total time.
"""
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

"""
    plot_cumulative_solve(allreports, args...; kwarg...)

Plot cumulative solver performance over time or steps for multiple simulation reports.

# Arguments
- `allreports`: Vector of simulation reports to analyze
- `args...`: Additional arguments passed to `plot_cumulative_solve!`

# Keyword Arguments
- `use_time = false`: Plot wall time instead of iterations
- `cumulative = true`: Show cumulative values vs individual step values
- `x_is_time = true`: Use time on x-axis instead of step numbers
- `legend = true`: Show legend for multiple datasets
- `scatter_points = true`: Add scatter points to the plot
- `linewidth = 3.5`: Line width for the plot
- Additional keyword arguments are passed to the plotting functions

# Returns
- `(fig, ax, alldata, t)`: Figure, axis, processed data, and time vectors

Creates a new figure and plots cumulative solver performance metrics, useful for
comparing simulation efficiency across different configurations or time periods.
"""
function Jutul.plot_cumulative_solve(allreports, arg...; kwarg...)
    fig = Figure()
    ax, alldata, t = Jutul.plot_cumulative_solve!(fig[1, 1], allreports, arg...; kwarg...)
    display(fig)
    return (fig, ax, alldata, t)
end

"""
    plot_cumulative_solve!(f, allreports, dt = nothing, names = nothing; kwarg...)

Mutating version of `plot_cumulative_solve` that plots into an existing figure layout.

# Arguments
- `f`: Figure or layout position to plot into
- `allreports`: Vector of simulation reports to analyze
- `dt = nothing`: Time step sizes (auto-detected if `nothing`)
- `names = nothing`: Names for each dataset (auto-generated if `nothing`)

# Keyword Arguments
- `use_time = false`: Plot wall time instead of iterations
- `use_title = true`: Show plot title
- `linewidth = 3.5`: Line width for the plot
- `cumulative = true`: Show cumulative values vs individual step values
- `linestyles = missing`: Custom line styles for each dataset
- `colormap = :tab20`: Colormap for multiple datasets
- `t_scale = ("s", 1.0)`: Time scale unit and conversion factor
- `x_is_time = true`: Use time on x-axis instead of step numbers
- `legend = true`: Show legend for multiple datasets
- `ministeps = false`: Include ministep information
- `title = nothing`: Custom plot title
- `scatter_points = true`: Add scatter points to the plot

# Returns
- `(ax, alldata, t)`: Axis object, processed data, and time vectors

This function provides fine-grained control over cumulative performance plotting
by allowing integration into custom figure layouts.
"""
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


"""
    plot_linear_convergence(report; kwarg...)

Plot the convergence history of linear solver iterations from a simulation report.

# Arguments
- `report`: Simulation report containing linear solver information

# Keyword Arguments
- Additional keyword arguments are passed to the `Axis` constructor

# Returns
- `fig`: Figure object containing the convergence plot

Creates a logarithmic plot showing how the linear solver residual decreases over
iterations. This is useful for analyzing linear solver performance and convergence
behavior during simulation.
"""
function Jutul.plot_linear_convergence(report; kwarg...)
    fig = Figure()
    ax = Axis(fig[1, 1], yscale = log10; ylabel = "Residual", xlabel = "Linear iterations", kwarg...)
    plot_linear_convergence!(ax, report)
    return fig
end

"""
    plot_linear_convergence!(ax, report)

Mutating version of `plot_linear_convergence` that plots into an existing axis.

# Arguments
- `ax`: Makie Axis object to plot into
- `report`: Simulation report or vector of reports containing linear solver information

This function extracts linear solver residual information from simulation reports
and plots the convergence history. It can handle individual reports, nested report
structures with ministeps, or vectors of multiple reports.
"""
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
