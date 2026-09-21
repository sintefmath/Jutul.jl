function variable_graph_levels(nodes, dependencies, node_index)
    n = length(nodes)
    levels = zeros(Int, n)
    incoming = [Int[] for _ in 1:n]
    outgoing = [Int[] for _ in 1:n]
    indegree = zeros(Int, n)
    for (target, deps) in enumerate(dependencies), dependency in deps
        source = node_index[dependency]
        push!(incoming[target], source)
        push!(outgoing[source], target)
        indegree[target] += 1
    end

    # Variable graphs are normally DAGs. Picking the first remaining node when
    # there is no root also gives model graphs with cyclic cross terms a stable
    # layout: the edge which closes a cycle is simply routed backwards later.
    remaining = trues(n)
    for _ in 1:n
        node = findfirst(i -> remaining[i] && indegree[i] == 0, 1:n)
        isnothing(node) && (node = findfirst(remaining))
        predecessors = filter(i -> !remaining[i], incoming[node])
        if !isempty(predecessors)
            levels[node] = maximum(levels[predecessors]) + 1
        end
        remaining[node] = false
        for target in outgoing[node]
            indegree[target] -= 1
        end
    end
    return levels
end

function update_layout_ranks!(ranks, layers)
    for layer in layers, (rank, node) in enumerate(layer)
        ranks[node] = rank
    end
    return ranks
end

function reorder_layout_layer!(layer, neighbors, ranks)
    old_rank = copy(ranks)
    sort!(layer; alg = MergeSort, by = node -> begin
        adjacent = neighbors[node]
        barycenter = isempty(adjacent) ? old_rank[node] :
            sum(old_rank[i] for i in adjacent)/length(adjacent)
        # The old rank makes ties deterministic and prevents unrelated nodes
        # from jumping around between sweeps.
        (barycenter, old_rank[node])
    end)
    return layer
end

function count_layout_crossings(layers, next_neighbors, ranks)
    crossings = 0
    for level in 1:(length(layers) - 1)
        segments = Tuple{Int, Int}[]
        for source in layers[level], target in next_neighbors[source]
            push!(segments, (source, target))
        end
        for i in eachindex(segments), j in (i + 1):length(segments)
            s1, t1 = segments[i]
            s2, t2 = segments[j]
            if s1 != s2 && t1 != t2 &&
                    (ranks[s1] - ranks[s2])*(ranks[t1] - ranks[t2]) < 0
                crossings += 1
            end
        end
    end
    return crossings
end

function minimize_layout_crossings!(layers, previous_neighbors, next_neighbors;
        sweeps = 8)
    ranks = zeros(Int, length(previous_neighbors))
    update_layout_ranks!(ranks, layers)
    best_layers = deepcopy(layers)
    best_crossings = count_layout_crossings(layers, next_neighbors, ranks)
    for _ in 1:sweeps
        for level in 2:length(layers)
            reorder_layout_layer!(layers[level], previous_neighbors, ranks)
            update_layout_ranks!(ranks, layers)
        end
        for level in (length(layers) - 1):-1:1
            reorder_layout_layer!(layers[level], next_neighbors, ranks)
            update_layout_ranks!(ranks, layers)
        end
        crossings = count_layout_crossings(layers, next_neighbors, ranks)
        if crossings < best_crossings
            best_crossings = crossings
            best_layers = deepcopy(layers)
        end
        iszero(best_crossings) && break
    end
    for i in eachindex(layers)
        layers[i] = best_layers[i]
    end
    return update_layout_ranks!(ranks, layers)
end

function align_layout_layer!(coordinates, layer, neighbors, row_gap;
        relaxation = 0.65)
    isempty(layer) && return coordinates
    targets = map(layer) do node
        adjacent = neighbors[node]
        if isempty(adjacent)
            coordinates[node]
        else
            neighbor_center = sum(coordinates[i] for i in adjacent)/length(adjacent)
            (1 - relaxation)*coordinates[node] + relaxation*neighbor_center
        end
    end

    # Project the desired coordinates onto the minimum-separation constraint.
    # After accounting for row_gap, this is one-dimensional isotonic
    # regression. It lets connected nodes line up without allowing overlap or
    # changing the crossing-minimized order of a layer.
    block_values = Float64[]
    block_weights = Int[]
    block_first = Int[]
    block_last = Int[]
    for i in eachindex(targets)
        value = -targets[i] - (i - 1)*row_gap
        push!(block_values, value)
        push!(block_weights, 1)
        push!(block_first, i)
        push!(block_last, i)
        while length(block_values) > 1 &&
                block_values[end - 1] > block_values[end]
            weight = block_weights[end - 1] + block_weights[end]
            value = (block_weights[end - 1]*block_values[end - 1] +
                block_weights[end]*block_values[end])/weight
            block_values[end - 1] = value
            block_weights[end - 1] = weight
            block_last[end - 1] = block_last[end]
            pop!(block_values)
            pop!(block_weights)
            pop!(block_first)
            pop!(block_last)
        end
    end
    for block in eachindex(block_values)
        for i in block_first[block]:block_last[block]
            coordinates[layer[i]] =
                -(block_values[block] + (i - 1)*row_gap)
        end
    end
    return coordinates
end

function layout_y_coordinates(layers, previous_neighbors, next_neighbors,
        row_gap; sweeps = 4)
    coordinates = zeros(Float64, length(previous_neighbors))
    for layer in layers
        offset = (length(layer) + 1)/2
        for (row, node) in enumerate(layer)
            coordinates[node] = row_gap*(offset - row)
        end
    end
    for _ in 1:sweeps
        for level in 2:length(layers)
            align_layout_layer!(coordinates, layers[level],
                previous_neighbors, row_gap)
        end
        for level in (length(layers) - 1):-1:1
            align_layout_layer!(coordinates, layers[level],
                next_neighbors, row_gap)
        end
    end
    if !isempty(coordinates)
        center = (minimum(coordinates) + maximum(coordinates))/2
        coordinates .-= center
    end
    return coordinates
end

function smooth_edge_path(path, iterations)
    smoothed = collect(path)
    for _ in 1:clamp(iterations, 0, 6)
        length(smoothed) < 2 && break
        refined = Makie.Point2f[first(smoothed)]
        for i in 1:(length(smoothed) - 1)
            p1, p2 = smoothed[i], smoothed[i + 1]
            push!(refined, Makie.Point2f(
                0.75p1[1] + 0.25p2[1], 0.75p1[2] + 0.25p2[2]))
            push!(refined, Makie.Point2f(
                0.25p1[1] + 0.75p2[1], 0.25p1[2] + 0.75p2[2]))
        end
        push!(refined, last(smoothed))
        smoothed = refined
    end
    return smoothed
end

function edge_arrow_geometry(path, level_gap)
    lengths = map(1:(length(path) - 1)) do i
        direction = path[i + 1] - path[i]
        sqrt(sum(abs2, direction))
    end
    total_length = sum(lengths)
    arrow_offset = min(0.25total_length, 0.3level_gap)
    arrow_distance = total_length - arrow_offset
    traversed = 0.0
    for (i, segment_length) in enumerate(lengths)
        if segment_length > 0 && traversed + segment_length >= arrow_distance
            direction = path[i + 1] - path[i]
            fraction = (arrow_distance - traversed)/segment_length
            return path[i] + fraction*direction, direction
        end
        traversed += segment_length
    end
    return last(path), last(path) - path[end - 1]
end

function variable_graph_layout(nodes, dependencies;
        level_gap = 3.0, row_gap = 1.0, sweeps = 8,
        return_paths = false)
    n = length(nodes)
    @assert length(dependencies) == n
    node_index = Dict(node => i for (i, node) in enumerate(nodes))
    levels = variable_graph_levels(nodes, dependencies, node_index)
    max_level = isempty(levels) ? 0 : maximum(levels)
    layers = [Int[] for _ in 0:max_level]
    for node in eachindex(nodes)
        push!(layers[levels[node] + 1], node)
    end

    # Replace the middle of a long edge by virtual nodes, one in every layer it
    # crosses. They participate in the ordering pass, keeping those edges away
    # from real nodes instead of drawing a diagonal through their labels.
    layout_levels = collect(levels)
    previous_neighbors = [Int[] for _ in 1:n]
    next_neighbors = [Int[] for _ in 1:n]
    edge_nodes = Vector{Int}[]
    backward_edges = Int[]
    for (target, deps) in enumerate(dependencies), dependency in deps
        source = node_index[dependency]
        route = [source]
        if levels[source] < levels[target]
            previous = source
            for level in (levels[source] + 1):(levels[target] - 1)
                virtual_node = length(layout_levels) + 1
                push!(layout_levels, level)
                push!(previous_neighbors, Int[])
                push!(next_neighbors, Int[])
                push!(layers[level + 1], virtual_node)
                push!(next_neighbors[previous], virtual_node)
                push!(previous_neighbors[virtual_node], previous)
                push!(route, virtual_node)
                previous = virtual_node
            end
            push!(next_neighbors[previous], target)
            push!(previous_neighbors[target], previous)
        else
            push!(backward_edges, length(edge_nodes) + 1)
        end
        push!(route, target)
        push!(edge_nodes, route)
    end

    minimize_layout_crossings!(layers, previous_neighbors,
        next_neighbors; sweeps = sweeps)
    y_coordinates = layout_y_coordinates(layers, previous_neighbors,
        next_neighbors, row_gap; sweeps = max(2, sweeps ÷ 2))
    layout_positions = Vector{Makie.Point2f}(undef, length(layout_levels))
    for (level, layer) in enumerate(layers)
        for node in layer
            layout_positions[node] = Makie.Point2f(
                level_gap*(level - 1), y_coordinates[node])
        end
    end

    edge_paths = [layout_positions[route] for route in edge_nodes]
    if !isempty(backward_edges)
        top = maximum(last, layout_positions) + row_gap
        for (lane, edge) in enumerate(backward_edges)
            source, target = first(edge_nodes[edge]), last(edge_nodes[edge])
            p1, p2 = layout_positions[source], layout_positions[target]
            if source == target
                # Give self-dependencies an actual loop instead of drawing the
                # same horizontal segment once in each direction.
                x = p1[1] + level_gap*(0.45 + 0.15lane)
                y = p1[2] + 0.6row_gap
                edge_paths[edge] = [p1, Makie.Point2f(x, p1[2]),
                    Makie.Point2f(x, y), Makie.Point2f(p1[1], y), p1]
            elseif levels[source] == levels[target]
                # Same-level dependencies use a vertical lane just to the
                # right, avoiding every node between the two endpoints.
                x = p1[1] + level_gap*(0.45 + 0.15lane)
                edge_paths[edge] = [p1, Makie.Point2f(x, p1[2]),
                    Makie.Point2f(x, p2[2]), p2]
            else
                # Feedback edges run in separate lanes above the graph. This
                # makes cycles visible without letting them cut through all
                # intervening nodes and forward edges.
                y = top + row_gap*(lane - 1)*0.6
                edge_paths[edge] = [p1,
                    Makie.Point2f(p1[1] + 0.35level_gap, p1[2]),
                    Makie.Point2f(p1[1] + 0.35level_gap, y),
                    Makie.Point2f(p2[1] - 0.35level_gap, y),
                    Makie.Point2f(p2[1] - 0.35level_gap, p2[2]), p2]
            end
        end
    end

    positions = layout_positions[1:n]
    if return_paths
        return positions, node_index, edge_paths
    else
        return positions, node_index
    end
end

function draw_jutul_graph(nodes, dependencies, colors;
        figure = (;), axis = (;), node_size = 20, edge_width = 3,
        edge_color = :grey60, text_size = 20, labels = string.(nodes),
        tooltips = nothing, implementations = nothing,
        node_padding = (8, 8, 4, 4), node_corner_radius = 5,
        text_color = :white, text_glow_color = (:black, 0.8),
        text_glow_width = 3, level_gap = 3.0, row_gap = 1.0,
        layout_sweeps = 8, edge_smoothing = 3)
    positions, node_index, edge_paths = variable_graph_layout(nodes,
        dependencies; level_gap = level_gap, row_gap = row_gap,
        sweeps = layout_sweeps, return_paths = true)
    draw_paths = [smooth_edge_path(path, edge_smoothing) for path in edge_paths]
    level_count = isempty(positions) ? 1 : length(unique(first.(positions)))
    longest_label = isempty(labels) ? 0 : maximum(length, labels)
    level_width = max(220, round(Int, 0.75text_size*longest_label))
    default_width = clamp(200 + level_count*level_width, 900, 2400)
    path_points = isempty(draw_paths) ? Makie.Point2f[] : vcat(draw_paths...)
    extent_points = vcat(positions, path_points)
    layout_y = last.(extent_points)
    yspan = isempty(layout_y) ? 0.0 : maximum(layout_y) - minimum(layout_y)
    default_height = clamp(375 + round(Int, 75yspan), 600, 1800)
    figure_options = merge((size = (default_width, default_height),), figure)
    fig = Makie.Figure(; figure_options...)
    ax = Makie.Axis(fig[1, 1]; axis...)

    edge = 0
    for (_, deps) in enumerate(dependencies), _ in deps
        edge += 1
        path = draw_paths[edge]
        Makie.lines!(ax, path; color = edge_color, linewidth = edge_width,
            linecap = :round, joinstyle = :round, inspectable = false)
        arrow_position, direction = edge_arrow_geometry(path, level_gap)
        angle = atan(direction[2], direction[1]) - pi/2
        Makie.scatter!(ax, [arrow_position]; marker = :utriangle,
            rotation = angle, markersize = 0.7node_size, color = edge_color,
            inspectable = false)
    end

    # Use a separate TextLabel recipe for each node. This lets both its rounded
    # background mesh and its text resolve unambiguously to one tooltip/click.
    plot_to_node = IdDict{Any, Int}()
    function register_node_plot!(plot, node_index)
        plot_to_node[plot] = node_index
        if hasproperty(plot, :plots)
            for child in plot.plots
                register_node_plot!(child, node_index)
            end
        end
    end
    function node_for_plot(plot)
        while !isnothing(plot)
            node_index = get(plot_to_node, plot, nothing)
            !isnothing(node_index) && return node_index
            plot = plot isa Makie.AbstractPlot ? plot.parent : nothing
        end
        return nothing
    end
    for i in eachindex(nodes)
        inspector_label = isnothing(tooltips) ? Makie.automatic :
            ((_, _, _) -> tooltips[i])
        inspector_hover = if isnothing(tooltips)
            Makie.automatic
        else
            (inspector, _, _, _...) -> begin
                mouse_position = Makie.mouseposition_px(inspector.root)
                Makie.update_tooltip_alignment!(inspector, mouse_position;
                    text = tooltips[i])
                return true
            end
        end
        node_plot = Makie.textlabel!(ax, positions[i]; text = labels[i],
            fontsize = text_size, padding = node_padding,
            background_color = colors[i], strokecolor = (:black, 0.75),
            strokewidth = 1.5, cornerradius = node_corner_radius,
            text_color = text_color, text_glowcolor = text_glow_color,
            text_glowwidth = text_glow_width,
            inspector_label = inspector_label,
            inspector_hover = inspector_hover)
        register_node_plot!(node_plot, i)
    end
    Makie.hidespines!(ax)
    Makie.hidedecorations!(ax)
    Makie.autolimits!(ax)

    # Makie does not include text extents in automatic axis limits. Add enough
    # data-space padding for the labels at the outermost levels and rows.
    xs = first.(extent_points)
    ys = last.(extent_points)
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
                node_index = node_for_plot(plot)
                if !isnothing(node_index)
                    edit_implementation(implementations[node_index])
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
    Makie.Legend(fig[1, 1],
        [Makie.MarkerElement(color = palette[i], marker = :rect,
            markersize = Makie.Vec2f(28, 16), strokewidth = 1) for i in 1:3],
        ["Primary variable", "Secondary variable", "Parameter"];
        orientation = :horizontal, tellheight = false, tellwidth = false,
        halign = :center, valign = :bottom, margin = (10, 10, 10, 10))
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
