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
        edge_color = :grey80, text_size = 20, labels = string.(nodes))
    positions, node_index = variable_graph_layout(nodes, dependencies)
    fig = Makie.Figure(; figure...)
    ax = Makie.Axis(fig[1, 1]; aspect = Makie.DataAspect(), axis...)

    for (target, deps) in enumerate(dependencies), dependency in deps
        source = node_index[dependency]
        p1, p2 = positions[source], positions[target]
        Makie.lines!(ax, [p1, p2]; color = edge_color, linewidth = edge_width)
        direction = p2 - p1
        angle = atan(direction[2], direction[1]) - pi/2
        Makie.scatter!(ax, [p2]; marker = :utriangle, rotation = angle,
            markersize = 0.65node_size, color = edge_color)
    end

    Makie.scatter!(ax, positions; markersize = node_size, color = colors,
        strokewidth = 1, strokecolor = :black)
    Makie.text!(ax, labels; position = positions, fontsize = text_size,
        align = (:center, :bottom), offset = (0, node_size/2 + 5))
    Makie.hidespines!(ax)
    Makie.hidedecorations!(ax)
    Makie.autolimits!(ax)
    return fig, ax
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
    fig, _ = draw_jutul_graph(nodes, dependencies, colors; kwargs...)
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
