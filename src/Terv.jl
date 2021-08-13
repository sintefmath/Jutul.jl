module Terv

# Use ForwardDiff.Duals
using ForwardDiff
# Some light type piracy to fix:
# https://github.com/JuliaDiff/ForwardDiff.jl/issues/542
export iszero
import ForwardDiff.Dual
import Base.iszero
Base.iszero(d::ForwardDiff.Dual) = false# iszero(d.value) && iszero(d.partials)

using LinearAlgebra
using BenchmarkTools
using SparseArrays
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

include("plot_graph.jl")
include("battery/battery_include.jl")

end # module
