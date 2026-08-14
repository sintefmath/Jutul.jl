"""
    Streamlines

Module for tracing streamlines through unstructured meshes.

Provides functionality for:
- Cell tesselation into tetrahedral/triangular sub-cells
- Velocity reconstruction from face fluxes
- Octree spatial indexing for fast point location
- Multiple integration methods (Euler, RK2, RK4)
- Forward and backward streamline tracing
"""
module Streamlines

using StaticArrays
using LinearAlgebra
using ..Jutul

export StreamlineTracer, setup_streamline_tracer, trace_streamlines
export EulerIntegrator, RK2Integrator, RK4Integrator

# Core types
include("types.jl")

# Integration methods
include("integrators.jl")

# Octree implementation
include("octree.jl")

# Tesselation and velocity reconstruction
include("tesselation.jl")

# Main tracing functionality
include("tracing.jl")

end # module
