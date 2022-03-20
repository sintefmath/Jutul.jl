module Jutul

# Use ForwardDiff.Duals
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
using DataStructures
using OrderedCollections
# Various bits and bobs from Base
using Statistics
using Logging
using Printf
using Dates
# Gotta go fast
using Tullio
using LoopVectorization
using CUDA, CUDAKernels
using KernelAbstractions
using Polyester
# Linear solvers and preconditioners

# Misc. utils
using MAT
using ExprTools
using LightGraphs
using PrettyTables
using Polynomials
using JLD2
# Nice progress bars
using ProgressMeter
# Timing
using TimerOutputs
# PVT
using MultiComponentFlash
# Conditional code loading
using Requires

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
include("timesteps.jl")
include("simulator.jl")

include("utils.jl")
include("interpolation.jl")
# 
include("multimodel/multimodel.jl")
include("dd/dd.jl")

# Various add-ons

# Reservoir simulator
include("applications/reservoir_simulator/reservoir_simulator.jl")
# Test systems
include("applications/test_systems/test_systems.jl")
# Battery/electrolyte
include("applications/battery/battery.jl")

include("meshes/meshes.jl")

# Plotting
function __init__()
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" begin
        @require ColorSchemes="35d6a980-a343-548e-a6ea-1d62b119f2f4" include("plotting.jl")
    end
    @require GraphRecipes="bd48cda9-67a9-57be-86fa-5b3c104eda73" begin
        @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot_graph.jl")
    end
end
end # module
