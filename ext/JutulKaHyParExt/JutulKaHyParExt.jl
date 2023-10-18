module JutulKaHyParExt
    using Jutul, KaHyPar
    function Jutul.partition_hypergraph_implementation(hg, n, mp::KaHyParPartitioner)
        G_khp = KaHyPar.HyperGraph(hg.graph, hg.node_weights, hg.edge_weights)
        return KaHyPar.partition(G_khp, n, configuration = mp.configuration)
    end
end
