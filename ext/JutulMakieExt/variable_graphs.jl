function variable_graph_layout(nodes, dependencies)
    n = length(nodes)
    node_index = Dict(node => i for (i, node) in enumerate(nodes))
    levels = zeros(Int, n)

    # Variables are already topologically sorted by Jutul. Iterating to a fixed
    # point also makes this helper robust to models constructed by hand.
    for _ in 1:n
        changed = false
        for (i, deps) in enumerate(dependencies)
            isempty(deps) && continue
            level = maximum(levels[node_index[d]] for d in deps) + 1
            if level > levels[i]
                levels[i] = level
                changed = true
            end
        end
        changed || break
    end

    positions = Vector{Makie.Point2f}(undef, n)
    for level in unique(levels)
        indices = findall(==(level), levels)
        offset = (length(indices) + 1)/2
        for (row, i) in enumerate(indices)
            positions[i] = Makie.Point2f(3level, offset - row)
        end
    end
    return positions, node_index
end

function draw_jutul_graph(nodes, dependencies, colors;
        figure = (;), axis = (;), node_size = 20, edge_width = 3,
        edge_color = :grey60, text_size = 20, labels = string.(nodes),
        tooltips = nothing, implementations = nothing)
    positions, node_index = variable_graph_layout(nodes, dependencies)
    level_count = isempty(positions) ? 1 : length(unique(first.(positions)))
    rows_per_level = isempty(positions) ? [1] :
        [count(==(x), first.(positions)) for x in unique(first.(positions))]
    longest_label = isempty(labels) ? 0 : maximum(length, labels)
    level_width = max(220, round(Int, 0.75text_size*longest_label))
    default_width = clamp(200 + level_count*level_width, 900, 2400)
    default_height = clamp(300 + 75maximum(rows_per_level), 600, 1600)
    figure_options = merge((size = (default_width, default_height),), figure)
    fig = Makie.Figure(; figure_options...)
    ax = Makie.Axis(fig[1, 1]; aspect = Makie.DataAspect(), axis...)

    for (target, deps) in enumerate(dependencies), dependency in deps
        source = node_index[dependency]
        p1, p2 = positions[source], positions[target]
        Makie.lines!(ax, [p1, p2]; color = edge_color, linewidth = edge_width,
            inspectable = false)
        direction = p2 - p1
        angle = atan(direction[2], direction[1]) - pi/2
        # Put the arrowhead within the edge so that the target node does not
        # cover it. This also makes direction visible on short diagonal edges.
        arrow_position = p1 + 0.65direction
        Makie.scatter!(ax, [arrow_position]; marker = :utriangle,
            rotation = angle, markersize = 0.7node_size, color = edge_color,
            inspectable = false)
    end

    inspector_label = isnothing(tooltips) ? Makie.automatic :
        ((_, index, _) -> tooltips[index])
    node_plot = Makie.scatter!(ax, positions; markersize = node_size,
        color = colors, strokewidth = 1, strokecolor = :black,
        inspector_label = inspector_label)
    Makie.text!(ax, labels; position = positions, fontsize = text_size,
        align = (:center, :bottom), offset = (0, node_size/2 + 5),
        inspectable = false)
    Makie.hidespines!(ax)
    Makie.hidedecorations!(ax)
    Makie.autolimits!(ax)

    # Makie does not include text extents in automatic axis limits. Add enough
    # data-space padding for the labels at the outermost levels and rows.
    xs = first.(positions)
    ys = last.(positions)
    xspan = isempty(xs) ? 0.0 : maximum(xs) - minimum(xs)
    yspan = isempty(ys) ? 0.0 : maximum(ys) - minimum(ys)
    xpad = max(3.0, 0.15xspan)
    ypad = max(1.5, 0.2yspan)
    isempty(xs) || Makie.xlims!(ax, minimum(xs) - xpad, maximum(xs) + xpad)
    isempty(ys) || Makie.ylims!(ax, minimum(ys) - ypad, maximum(ys) + ypad)

    if !isnothing(tooltips)
        Makie.DataInspector(fig)
    end
    if !isnothing(implementations) && Jutul.plotting_check_interactive(warn = false)
        Makie.on(Makie.events(fig).mousebutton, priority = 2) do event
            if event.button == Makie.Mouse.left && event.action == Makie.Mouse.press
                plot, index = Makie.pick(fig)
                if plot == node_plot && 1 <= index <= length(implementations)
                    edit_implementation(implementations[index])
                    return Makie.Consume(true)
                end
            end
            return Makie.Consume(false)
        end
    end
    return fig, ax
end

implementation_constructor(implementation) =
    Base.typename(typeof(implementation)).wrapper

function edit_implementation(implementation)
    # Resolve constructors through the type wrapper so all parameterizations
    # of e.g. Variable{T, N} navigate to the same constructor definitions.
    constructor = implementation_constructor(implementation)
    candidates = methods(constructor).ms
    method_index = findfirst(candidates) do method
        method.line > 0 && !(string(method.file) in ("boot.jl", "none"))
    end
    if isnothing(method_index)
        @warn "Could not find a source location for $constructor"
    else
        try
            # InteractiveUtils is loaded in Main on demand so that plotting
            # retains Makie as its only package dependency.
            Base.require(Main, :InteractiveUtils)
            Base.invokelatest(Main.InteractiveUtils.edit,
                candidates[method_index])
        catch exception
            @warn "Could not open the constructor for $constructor" exception
        end
    end
    return nothing
end

function variable_implementation(model, node)
    for collection in (model.primary_variables, model.secondary_variables,
            model.parameters)
        haskey(collection, node) && return collection[node]
    end
    return nothing
end

function Jutul.plot_variable_graph(model; kwargs...)
    nodes, dependencies = Jutul.build_variable_graph(model)
    palette = Makie.wong_colors()
    colors = map(nodes) do node
        if haskey(model.primary_variables, node)
            palette[1]
        elseif haskey(model.secondary_variables, node)
            palette[2]
        elseif haskey(model.parameters, node)
            palette[3]
        else
            :black
        end
    end
    implementations = [variable_implementation(model, node) for node in nodes]
    tooltips = ["$(nodes[i])\n$(typeof(implementations[i]))" for i in eachindex(nodes)]
    fig, _ = draw_jutul_graph(nodes, dependencies, colors;
        tooltips = tooltips, implementations = implementations, kwargs...)
    Makie.Legend(fig[2, 1],
        [Makie.MarkerElement(color = palette[i], marker = :circle,
            markersize = 20, strokewidth = 1) for i in 1:3],
        ["Primary variable", "Secondary variable", "Parameter"];
        orientation = :horizontal)
    return fig
end

Jutul.plot_model_graph(model; kwargs...) = Jutul.plot_variable_graph(model; kwargs...)

function Jutul.plot_model_graph(model::Jutul.MultiModel; kwargs...)
    nodes = Symbol[]
    dependencies = Vector{Symbol}[]
    colors = Any[]
    palette = Makie.wong_colors()

    model_nodes = Dict{Any, Symbol}()
    equation_nodes = Dict{Tuple{Any, Any}, Symbol}()
    for (model_key, submodel) in pairs(model.models)
        model_node = Symbol(model_key)
        model_nodes[model_key] = model_node
        push!(nodes, model_node)
        push!(dependencies, Symbol[])
        push!(colors, palette[1])
        for equation_key in keys(submodel.equations)
            equation_node = Symbol(model_key, "__", equation_key)
            equation_nodes[(model_key, equation_key)] = equation_node
            push!(nodes, equation_node)
            push!(dependencies, [model_node])
            push!(colors, palette[2])
        end
    end

    for pair in model.cross_terms
        (; target, source, target_equation, source_equation) = pair
        push!(dependencies[findfirst(==(model_nodes[source]), nodes)],
            equation_nodes[(target, target_equation)])
        if Jutul.has_symmetry(pair.cross_term)
            push!(dependencies[findfirst(==(model_nodes[target]), nodes)],
                equation_nodes[(source, source_equation)])
        end
    end
    labels = map(nodes) do node
        parts = split(string(node), "__"; limit = 2)
        last(parts)
    end
    fig, _ = draw_jutul_graph(nodes, dependencies, colors;
        labels = labels, kwargs...)
    return fig
end
