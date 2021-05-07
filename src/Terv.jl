module Terv

using SparseArrays
using LinearAlgebra
using BenchmarkTools
using ForwardDiff
using KernelAbstractions, CUDA, CUDAKernels
using Logging
# Main types
include("core_types.jl")

# MRST stuff
include("mrst_input.jl")
# Grids, types
include("grids.jl")
# Meat and potatoes
include("static_structures.jl")
include("assembly.jl")
include("benchmarks.jl")
include("linsolve.jl")

include("equation_helpers.jl")


include("multiphase.jl")

include("porousmedia.jl")

include("simulator.jl")
end # module
