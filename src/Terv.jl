module Terv

using SparseArrays
using LinearAlgebra
using BenchmarkTools
using ForwardDiff
using KernelAbstractions, CUDA, CUDAKernels

# MRST stuff
include("mrst_input.jl")
# Grids, types
include("grids.jl")
# Meat and potatoes
include("static_structures.jl")
include("assembly.jl")
include("benchmarks.jl")
include("linsolve.jl")
include("formulations.jl")
include("models.jl")

include("core_types.jl")
include("multiphase.jl")

include("porousmedia.jl")
include("equation_helpers.jl")
end # module
