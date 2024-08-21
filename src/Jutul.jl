module Jutul
    using ForwardDiff
    # Arrays etc
    using LinearAlgebra
    using SparseArrays
    using MappedArrays
    using StaticArrays
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
    using Polyester
    using PolyesterWeave
    # Linear solvers and preconditioners
    using ILUZero
    using LinearOperators
    using Krylov
    using AlgebraicMultigrid

    # Misc. utils
    using ExprTools
    using Graphs
    using PrettyTables
    using Polynomials
    using JLD2, MAT

    import Metis
    import SymRCM
    # Nice progress bars
    using ProgressMeter
    using Crayons
    using Crayons.Box

    # Timing
    using TimerOutputs
    # Shorted alias for @timeit_debug
    const var"@tic" = var"@timeit_debug"
    const JUTUL_OUTPUT_TYPE = OrderedDict{Symbol, Any}

    timeit_debug_enabled() = false

    # Separate module for CSR backend
    include("StaticCSR/StaticCSR.jl")
    using .StaticCSR
    import .StaticCSR: nthreads, minbatch
    import SparsityTracing as ST

    # Main types
    include("core_types/core_types.jl")

    # Models 
    include("models.jl")

    # MRST stuff
    # Grids, types
    include("domains.jl")

    # Meat and potatoes
    include("variable_evaluation.jl")
    include("conservation/flux.jl")
    include("conservation/fvm_assembly.jl")
    include("linsolve/linsolve.jl")

    include("context.jl")
    include("equations.jl")
    include("ad/ad.jl")
    include("variables/variables.jl")

    include("conservation/conservation.jl")
    include("timesteps.jl")
    include("simulator/simulator.jl")

    include("utils.jl")
    include("config.jl")
    include("interpolation.jl")
    include("partitioning.jl")

    # Systems that are made up of subsystems
    include("composite/composite.jl")
    # Models that contain submodels
    include("multimodel/multimodel.jl")
    # Domain decomposition
    include("dd/dd.jl")

    # Test systems
    include("applications/test_systems/test_systems.jl")

    include("meshes/meshes.jl")
    include("discretization/discretization.jl")

    # Extensions' interfaces
    include("ext/extensions.jl")

    # Support for SI unit conversion
    include("units/units.jl")

    # Nonlinear finite-volume discretizations
    include("NFVM/NFVM.jl")

end # module
