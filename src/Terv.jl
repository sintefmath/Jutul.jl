module Terv

using Base: Symbol, Real
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using ForwardDiff
using KernelAbstractions, CUDA, CUDAKernels
using Logging
using MappedArrays
using Printf
using Dates
using DataStructures
using Tullio
using PrettyTables

# Main types
include("core_types.jl")

# Models 
include("models.jl")
include("multimodel.jl")

# include("models.jl")
# MRST stuff
# Grids, types
include("domains.jl")

# Meat and potatoes
include("variable_evaluation.jl")
include("conservation/flux.jl")
include("linsolve.jl")

include("context.jl")
include("equations.jl")
include("ad.jl")
include("variables.jl")

include("conservation/conservation.jl")
include("simulator.jl")

include("utils.jl")

# Various add-ons
include("applications/reservoir_simulator/reservoir_simulator.jl")
include("applications/test_systems/test_systems.jl")

include("battery/battery_types.jl")
include("battery/battery.jl")
include("battery/test_setup.jl")

end # module
