module AdjointsDI
    using Jutul
    import Jutul: @tic
    using DifferentiationInterface
    using SparseConnectivityTracer
    using SparseMatrixColorings

    timeit_debug_enabled() = Jutul.timeit_debug_enabled()

    include("adjoints.jl")
    include("utils.jl")
end
