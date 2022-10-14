module Jutul
    using ForwardDiff
    # Some light type piracy to fix:
    # https://github.com/JuliaDiff/ForwardDiff.jl/issues/542
    export iszero
    import ForwardDiff.Dual
    import Base.iszero
    Base.iszero(d::ForwardDiff.Dual) = false# iszero(d.value) && iszero(d.partials)

    # Arrays etc
    using LinearAlgebra
    using SparseArrays
    using MappedArrays
    # Data structures
    import DataStructures: OrderedDict
    using OrderedCollections
    # Various bits and bobs from Base
    using Statistics
    using Logging
    using Printf
    using Dates
    # Gotta go fast
    using Tullio
    using LoopVectorization
    # using CUDA, CUDAKernels
    using KernelAbstractions
    using Polyester
    # Linear solvers and preconditioners

    # Misc. utils
    using ExprTools
    using Graphs
    using PrettyTables
    using Polynomials
    using JLD2

    import Metis
    # Nice progress bars
    using ProgressMeter
    using Crayons
    using Crayons.Box

    # Timing
    using TimerOutputs

    include("StaticCSR/StaticCSR.jl")
    using .StaticCSR
    import .StaticCSR: nthreads, minbatch
    import SparsityTracing as ST

    # Main types
    include("core_types.jl")

    # Models 
    include("models.jl")

    # include("models.jl")
    # MRST stuff
    # Grids, types
    include("domains.jl")

    # Meat and potatoes
    include("variable_evaluation.jl")
    include("conservation/flux.jl")
    include("linsolve/linsolve.jl")

    include("context.jl")
    include("equations.jl")
    include("ad/ad.jl")
    include("variables/variables.jl")

    include("conservation/conservation.jl")
    include("timesteps.jl")
    include("simulator/simulator.jl")

    include("utils.jl")
    include("interpolation.jl")
    include("partitioning.jl")
    # 
    include("multimodel/multimodel.jl")
    include("dd/dd.jl")

    # Test systems
    include("applications/test_systems/test_systems.jl")

    include("meshes/meshes.jl")
    include("discretization/discretization.jl")
end # module
