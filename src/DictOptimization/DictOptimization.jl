module DictOptimization
    using Jutul, PrettyTables
    include("types.jl")
    include("interface.jl")
    include("validation.jl")
    include("optimization.jl")
    include("utils.jl")
    include("scaler.jl")
    include("uq.jl")
end
