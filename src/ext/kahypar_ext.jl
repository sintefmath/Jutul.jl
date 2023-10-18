export KaHyParPartitioner

struct KaHyParPartitioner <: JutulPartitioner
    configuration::Symbol
    function KaHyParPartitioner(t = :edge_cut)
        @assert t == :connectivity || t == :edge_cut
        return new(t)
    end
end
