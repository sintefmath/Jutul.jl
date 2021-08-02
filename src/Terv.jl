module Terv

using SparseArrays
using LinearAlgebra
using BenchmarkTools
using ForwardDiff
using KernelAbstractions, CUDA, CUDAKernels
using Logging
using MappedArrays
using Printf
using Dates
using DataStructures, OrderedCollections
using LoopVectorization
using Tullio
using PrettyTables
using DataInterpolations
using ILUZero

using Base.Threads
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
include("ad.jl")
include("variables.jl")

include("conservation/conservation.jl")
include("simulator.jl")

include("utils.jl")
include("interpolation.jl")
# 
include("multimodel/multimodel.jl")

# Various add-ons
include("applications/reservoir_simulator/reservoir_simulator.jl")
include("applications/test_systems/test_systems.jl")

include("battery/battery_types.jl")
include("battery/physical_constants.jl")
include("battery/tensor_tools.jl")
include("battery/elchem_component.jl")
include("battery/physics.jl")
include("battery/battery.jl")
include("battery/test_setup.jl")
include("battery/elyte.jl")
include("battery/current_collector.jl")

include("plot_graph.jl")

end # module
