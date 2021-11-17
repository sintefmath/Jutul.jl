using Plots, GraphRecipes

export plot_graph


function get_graph(model)
    primary = model.primary_variables
    secondary = model.secondary_variables
    nodes, edges = build_variable_graph(model, primary, secondary)
    return nodes, edges
end

function get_index_graph(nodes, edges)
    nodes_num = 1:size(nodes)[1]
    nodes_ind = Dict([nodes[i] => i for i in nodes_num])
    edges_ind = Vector{Vector{Any}}()

    for edges_vec in edges
        v = []
        for d in edges_vec
            push!(v, nodes_ind[d])
        end
        push!(edges_ind, v)
    end
    
    return edges_ind
end

function plot_graph(model)
    # nodes = [:a, :b, :c, :d]
    # edges = [[], [:a], [:a], [:a, :c]]

    nodes, edges = get_graph(model)
    nodes_name = [String(node) for node in nodes]
    edges_ind = get_index_graph(nodes, edges)

    p = graphplot(edges_ind, names=nodes_name, nodeshape=:rect, curvature_scalar=0.001)
    Plots.plot!(size=(1200, 1200))
    Plots.plot!(show=true)
end

