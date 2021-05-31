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
using DataStructures

# Main types
include("core_types.jl")

# Models 
include("models.jl")
include("multimodel.jl")

# include("models.jl")
# MRST stuff
include("mrst_input.jl")
# Grids, types
include("domains.jl")
include("porousmedia_grids.jl")

# Meat and potatoes
include("statefunctions.jl")
include("static_structures.jl")
include("flux.jl")
include("benchmarks.jl")
include("linsolve.jl")

include("context.jl")
include("equations.jl")
include("ad.jl")

include("equations/conservation.jl")
include("multiphase.jl")
include("multiphase_sf.jl")
include("wells.jl")
include("porousmedia.jl")

include("simulator.jl")

include("utils.jl")
# Various add-ons
include("test_systems.jl")
end # module
