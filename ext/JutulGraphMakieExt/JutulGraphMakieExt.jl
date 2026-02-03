module JutulGraphMakieExt

    using Jutul, Makie
    using GraphMakie, Graphs, LayeredLayouts, NetworkLayout

    function Jutul.plot_variable_graph(model)
        graph, nodes, = build_variable_graph(model, to_graph = true)
        c1 = Makie.ColorSchemes.tab10[1]
        c2 = Makie.ColorSchemes.tab10[2]
        c3 = Makie.ColorSchemes.tab10[3]

        colors = Vector()
        for n in nodes
            if n in keys(model.primary_variables)
                c = c1
            elseif n in keys(model.secondary_variables)
                c = c2
            elseif n in keys(model.parameters)
                c = c3
            else
                c = :black
            end
            push!(colors, c)
        end
        fig, ax, plt = plot_jutul_graph(graph, nodes, colors)
        ax.xminorticksvisible[] = false
        hidespines!(ax)
        hidedecorations!(ax)
        marg = (marker = :circle, markersize = 20, strokewidth = 1)
        Legend(fig[2, 1],
            [
                MarkerElement(color = c1; marg...),
                MarkerElement(color = c2; marg...),
                MarkerElement(color = c3; marg...)
            ],
            ["Primary variable", "Secondary variable", "Parameter"],
            orientation = :horizontal
        )
        autolimits!(ax)
        xmin, xmax = ax.xaxis.attributes.limits[]
        xlims!(ax, xmin-3, xmax+2)
        return fig
    end

    function plot_jutul_graph(graph, nodes, colors = [:black for _ in nodes]; kwarg...)
        xs, ys, paths = solve_positions(Zarate(), graph)
        lay = _ -> Point.(zip(xs,ys))
        # create a vector of Point2f per edge
        wp = [Point2f.(zip(paths[e]...)) for e in Graphs.edges(graph)]
        alignments = []
        for i in xs
            if i == 1
                al = (:right, :center)
            else
                al = (:center, :bottom)
            end
            push!(alignments, al)
        end
        N = length(nodes)
        node_size = 20

        return graphplot(graph,
            axis = (xminorticksvisible = false,),
            layout=lay,
            waypoints=wp,
            nlabels_distance=10,
            nlabels_textsize=20,
            arrow_size = 20,
            node_size = [node_size for i in 1:N],
            edge_width = [3 for i in 1:ne(graph)],
            edge_color = :grey80,
            node_color = colors,
            node_strokewidth=1,
            nlabels_align = alignments,
            nlabels = map(String, nodes)
        )
    end

    function Jutul.plot_model_graph(model; kwarg...)
        Jutul.plot_variable_graph(model; kwarg...)
    end

    function Jutul.plot_model_graph(model::MultiModel)
        equation_to_label(k, eq_name) = "$k: $eq_name"
        edges = Vector{String}()
        node_labels = Vector{String}()
        nodes = Vector{String}()
        node_colors = Vector{Float64}()

        for (k, m) in pairs(model.models)
            push!(nodes, "$k")
            push!(node_labels, "$k")
            push!(node_colors, 1.0)
            for (e, eq) in m.equations
                push!(nodes, equation_to_label(k, e))
                push!(node_labels, "$e")
                push!(node_colors, 0.0)
            end
        end

        n = length(nodes)
        directed = true
        if directed
            graph = SimpleDiGraph(n)
        else
            graph = SimpleGraph(n)
        end
        to_index(k, eq) = findfirst(isequal(equation_to_label(k, eq)), nodes)
        to_index(k) = findfirst(isequal("$k"), nodes)

        for (k, m) in pairs(model.models)
            for (e, eq) in m.equations
                T = to_index(k)
                F = to_index(k, e)
                if !has_edge(graph, T, F)
                    add_edge!(graph, T, F)
                    push!(edges, "")
                end
            end
        end

        function add_cross_term_edge!(ct, target, source, equation)
            T = to_index(target, equation)
            F = to_index(source)
            ct_bare_type = Base.typename(typeof(ct)).name
            if !has_edge(graph, T, F)
                add_edge!(graph, T, F)
                push!(edges, "$ct_bare_type"[1:end-2])
            end
        end

        for ctp in model.cross_terms
            (; cross_term, target, source, target_equation, source_equation) = ctp
            add_cross_term_edge!(cross_term, target, source, target_equation)
            if Jutul.has_symmetry(cross_term) && directed
                add_cross_term_edge!(cross_term, source, target, source_equation)
            end
        end
        layout = SFDP(Ptype=Float32, tol=0.01, C=0.01, K=1)
        layout = Shell()
        # layout = SquareGrid()
        # layout = Spectral()
        layout = Spring()
        return graphplot(graph, nlabels = node_labels,
                                elabels = edges,
                                nlabels_distance=15,
                                nlabels_textsize=20,
                                node_size = [30 for i in 1:n],
                                node_color = node_colors,
                                layout=layout
                                )
    end
end
